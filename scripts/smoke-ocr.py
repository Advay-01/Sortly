"""Check JPEG and scanned PDF OCR through a frozen service and local fixture LLM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    seen = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            files = json.loads(body["messages"][1]["content"])["files"]
            seen.extend(files)
            reply = {"files": [{"id": item["id"], "category": "Study",
                     "subcategory": "Physics", "confidence": "high",
                     "reasoning": "Recognized lecture text", "evidence_quote": "Physics Lecture Notes",
                     "filename_conflict": "none", "suggested_name": "", "name_evidence": ""}
                    for item in files]}
            encoded = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="sortly-frozen-ocr-") as temp:
        root = Path(temp)
        samples = root / "samples"
        samples.mkdir()
        image = Image.new("RGB", (1400, 600), "white")
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
        ImageDraw.Draw(image).multiline_text((60, 80),
            "Physics Lecture Notes\nThermodynamics and Heat Transfer\nTemperature and energy conservation",
            font=font, fill="black", spacing=30)
        image.save(samples / "scan.jpeg")
        image.save(samples / "scan.pdf", "PDF")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        token = secrets.token_hex(24)
        env = dict(os.environ, ORGANIZER_HOME=str(root / "state"),
                   ORGANIZER_CACHE_DB=str(root / "cache.db"), ORGANIZER_RECENT_SECONDS="0",
                   PYTHON_DOTENV_DISABLED="1", SORTLY_SHUTDOWN_TOKEN=token)
        base = f"http://127.0.0.1:{port}"
        with (root / "service.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(args.executable.resolve()), "--port", str(port)],
                cwd=root, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.monotonic() + 75
                while time.monotonic() < deadline:
                    try:
                        if requests.get(base + "/api/health", timeout=1).ok:
                            break
                    except requests.RequestException:
                        pass
                    if process.poll() is not None:
                        raise RuntimeError((root / "service.log").read_text())
                    time.sleep(.2)
                response = requests.post(base + "/api/providers/config", json={
                    "primary": "openai", "fallbacks": [], "connections": {"openai": {
                    "model": "fixture", "base_url": f"http://127.0.0.1:{provider.server_port}/v1",
                    "api_key": "fixture-key"}}}, timeout=15)
                response.raise_for_status()
                response = requests.post(base + "/api/preview", json={"source": str(samples),
                    "models": ["openai/fixture"]}, timeout=120)
                response.raise_for_status()
                files = response.json()["files"]
                assert len(files) == 2, files
                for file in files:
                    assert "ocr" in file["evidence_sources"], file
                    assert "Physics Lecture Notes" in file["evidence_excerpt"], file
                    assert file["target_path"] == "Study/Physics", file
                assert seen
                assert all("Thermodynamics" in item["content"] for item in seen), seen
                print("Frozen JPEG and scanned PDF OCR passed; fixture LLM only, no files moved.")
            finally:
                process.terminate()
                process.wait(timeout=15)
                provider.shutdown()


if __name__ == "__main__":
    main()
