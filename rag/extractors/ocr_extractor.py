"""OCR extractor for scanned PDFs via ocrmypdf + PyMuPDF."""

from __future__ import annotations

import os
import tempfile

import fitz  # PyMuPDF


def extract_ocr(
    filepath: str,
    *,
    language: str = "eng+chi_sim+chi_tra",
) -> tuple[str | None, dict, bool, str | None]:
    """Run OCR on a scanned PDF and extract text.

    Uses ocrmypdf to create a text-searchable PDF in a temp directory,
    then reads text via PyMuPDF.

    Returns (text, metadata, succeeded, error_message).
    """
    try:
        import ocrmypdf
    except ImportError:
        return None, {"page_count": 0}, False, "ocrmypdf not installed"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_output = os.path.join(tmpdir, "ocr_output.pdf")

            ocrmypdf.ocr(
                filepath,
                ocr_output,
                language=language,
                deskew=True,
                force_perform=True,
                progress_bar=False,
            )

            doc = fitz.open(ocr_output)
            page_count = doc.page_count
            text_parts = []

            for i in range(doc.page_count):
                page = doc.load_page(i)
                text_parts.append(page.get_text("text"))

            doc.close()

            full_text = "\n\n".join(text_parts)
            meta = {"page_count": page_count, "ocr_language": language}
            return full_text, meta, True, None

    except Exception as e:
        # Get page count even if OCR fails, for reporting
        page_count = None
        try:
            doc = fitz.open(filepath)
            page_count = doc.page_count
            doc.close()
        except Exception:
            pass
        return None, {"page_count": page_count}, False, str(e)
