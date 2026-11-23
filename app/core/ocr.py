"""Offline OCR using wheel-bundled models and PDFium; no model downloads."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

MAX_IMAGE_EDGE = 2400
MAX_PDF_PAGES = 2
_engine = None
_engine_lock = Lock()
# PDFium's native API is not thread safe, including across documents.
_pdf_lock = Lock()


def recognize_image(image, max_chars: int) -> str:
    from PIL import Image, ImageOps
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    global _engine
    if max_chars <= 0:
        return ""
    image = ImageOps.exif_transpose(image)
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    # Flatten transparency onto white, then convert RGB to OpenCV's BGR order.
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    rgb = Image.alpha_composite(background, rgba).convert("RGB")
    pixels = np.asarray(rgb)[:, :, ::-1].copy()
    with _engine_lock:
        if _engine is None:
            _engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        results, _ = _engine(pixels)
    return " ".join(row[1].strip() for row in (results or []) if row[2] >= 0.5)[:max_chars]


def image_text(path: Path, max_chars: int) -> str:
    from PIL import Image

    with Image.open(path) as image:
        return recognize_image(image, max_chars)


def pdf_text(path: Path, max_chars: int) -> tuple[str, bool]:
    """Read the first two pages, rasterizing only pages without useful text."""
    import pypdfium2 as pdfium

    parts = []
    used = False
    with _pdf_lock:
        with pdfium.PdfDocument(str(path)) as document:
            for index in range(min(len(document), MAX_PDF_PAGES)):
                remaining = max_chars - len(" ".join(parts))
                if remaining <= 0:
                    break
                page = document[index]
                try:
                    textpage = page.get_textpage()
                    try:
                        text = " ".join(textpage.get_text_bounded().split())
                    finally:
                        textpage.close()
                    if len(text) < 20:
                        width, height = page.get_size()
                        scale = min(2.5, MAX_IMAGE_EDGE / max(width, height, 1))
                        bitmap = page.render(scale=scale)
                        try:
                            with bitmap.to_pil() as image:
                                recognized = recognize_image(image, remaining)
                        finally:
                            bitmap.close()
                        if recognized:
                            text = recognized
                            used = True
                    if text:
                        parts.append(text[:remaining])
                finally:
                    page.close()
    return " ".join(parts)[:max_chars], used
