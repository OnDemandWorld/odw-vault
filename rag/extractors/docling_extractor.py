"""Docling-based extractor for PDF, DOCX, PPTX, XLSX, HTML."""

from __future__ import annotations


def extract_docling(filepath: str) -> tuple[str | None, dict, bool, str | None]:
    """Convert a document file to markdown text via Docling.

    Returns (text, metadata, succeeded, error_message).
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(filepath)

        md_text = result.document.export_to_markdown()

        meta: dict = {
            "page_count": 0,
            "table_count": 0,
        }

        # Extract page count from the document if available
        doc = result.document
        if hasattr(doc, "pages"):
            meta["page_count"] = len(doc.pages)

        # Count tables in the markdown
        if md_text:
            meta["table_count"] = md_text.count("|") // 2
            # Try to get title from first heading
            first_line = md_text.split("\n")[0].strip()
            if first_line.startswith("# "):
                meta["title"] = first_line[2:].strip()

        return md_text, meta, True, None

    except ImportError:
        return None, {}, False, "docling not installed"
    except Exception as e:
        return None, {}, False, str(e)
