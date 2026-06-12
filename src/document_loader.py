import io
import PyPDF2


def _read_pdf_content(uploaded_file) -> str:
    """Parse a PDF file object and extract all page text."""
    page_texts = []
    try:
        raw_bytes  = io.BytesIO(uploaded_file.read())
        pdf_reader = PyPDF2.PdfReader(raw_bytes)
        for pg in pdf_reader.pages:
            extracted = pg.extract_text()
            if extracted:
                page_texts.append(extracted)
    except Exception as exc:
        return f"[PDF read error: {exc}]"
    return "\n".join(page_texts)


def _read_txt_content(uploaded_file) -> str:
    """Decode a plain-text file object to a Python string."""
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return f"[Text read error: {exc}]"


def load_uploaded_files(file_list) -> list:
    """
    Accept a list of Streamlit UploadedFile objects and return
    a list of dicts, each with keys 'name' and 'text'.
    Only non-empty documents are included in the output.
    """
    result = []
    for item in file_list:
        content = (
            _read_pdf_content(item)
            if item.name.lower().endswith(".pdf")
            else _read_txt_content(item)
        )
        if content.strip():
            result.append({"name": item.name, "text": content})
    return result
