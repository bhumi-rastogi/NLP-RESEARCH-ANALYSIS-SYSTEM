# reads PDF and TXT files and return raw text.

from pathlib import Path
from PyPDF2 import PdfReader


# Extract all text from a PDF file.
def extract_text_from_pdf(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


# Read all text from a plain .txt file.
def extract_text_from_txt(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    return file_path.read_text(encoding="utf-8", errors="replace")


# Ingest one or more PDF/TXT files and return their combined text.
def ingest_files(file_paths):
    texts = []
    for fp in file_paths:
        fp = Path(fp)
        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            texts.append(extract_text_from_pdf(fp))
        elif suffix == ".txt":
            texts.append(extract_text_from_txt(fp))
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}' for '{fp}'. "
                "Only .pdf and .txt files are accepted."
            )
    return "\n\n".join(texts)


# Extract text from a list of PDF files.
def extract_text_from_multiple_pdfs(file_paths):
    return [extract_text_from_pdf(fp) for fp in file_paths]
