"""Exercise real offline recognition, PDF rendering and evidence privacy."""
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.core.evidence import extract_evidence
from app.core.ocr import image_text, pdf_text
from app.models.schemas import FileInfo


def info(path):
    return FileInfo(path=str(path), filename=path.name, extension=path.suffix,
                    size_kb=path.stat().st_size / 1024, date_created="2026-11-23")


def scan(text):
    image = Image.new("RGB", (1400, 600), "white")
    fonts = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]
    font_path = next((path for path in fonts if Path(path).is_file()), None)
    if not font_path:
        pytest.skip("No fixture font available")
    ImageDraw.Draw(image).multiline_text((60, 80), text, fill="black",
                                        font=ImageFont.truetype(font_path, 42), spacing=30)
    return image


def test_real_jpeg_and_image_only_pdf_need_no_external_tools(tmp_path):
    image = scan("Physics Lecture Notes\nThermodynamics and Heat Transfer\nTemperature and energy conservation")
    jpeg = tmp_path / "scan.jpeg"
    pdf = tmp_path / "scan.pdf"
    image.save(jpeg)
    image.save(pdf, "PDF")
    before = pdf.read_bytes()
    with patch("app.core.evidence.shutil.which", return_value=None):
        for path in (jpeg, pdf):
            evidence = extract_evidence(info(path))
            assert "Thermodynamics" in evidence.content
            assert evidence.ocr_used
            assert "ocr" in evidence.sources
            assert evidence.extraction_status == "sufficient"
            assert not evidence.extension_mismatch
    assert pdf.read_bytes() == before
    assert len(image_text(jpeg, 25)) <= 25


def test_pdf_ocr_page_and_text_budgets(tmp_path):
    pdf = tmp_path / "three-pages.pdf"
    first = scan("Physics Lecture Notes")
    first.save(pdf, "PDF", save_all=True,
               append_images=[scan("Chemistry Laboratory Notes"), scan("Hidden Third Page")])
    text, used = pdf_text(pdf, 4000)
    assert used
    assert "Physics" in text and "Chemistry" in text
    assert "Hidden" not in text
    assert len(pdf_text(pdf, 10)[0]) <= 10


def test_jpeg_signature_and_ocr_privacy_checks(tmp_path):
    path = tmp_path / "misleading.bin"
    Image.new("RGB", (32, 32), "white").save(path, "JPEG")
    with patch("app.core.evidence._ocr_image", return_value=("Passport number: A1234567", True)):
        evidence = extract_evidence(info(path))
    assert evidence.ocr_used and evidence.extension_mismatch
    assert evidence.sensitive
    assert "A1234567" not in evidence.redacted_content


def test_ocr_failure_does_not_abort_extraction(tmp_path):
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (32, 32), "white").save(path)
    with (patch("app.core.ocr.image_text", side_effect=RuntimeError("unavailable")),
          patch("app.core.evidence.shutil.which", return_value=None)):
        evidence = extract_evidence(info(path))
    assert not evidence.ocr_used
    assert evidence.extraction_status == "unavailable"
    assert any("could not run" in warning for warning in evidence.warnings)
