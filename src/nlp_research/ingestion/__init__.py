"""Ingestion sub-package: PDF and TXT text extraction."""
from .pdf_extractor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_multiple_pdfs,
    ingest_files,
)

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "extract_text_from_multiple_pdfs",
    "ingest_files",
]
