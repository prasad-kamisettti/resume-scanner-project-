import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Existing path-based extractor (kept for test.py)."""
    text = ""
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_pdf_bytes(data: bytes) -> str:
    """New: in-memory extractor for Streamlit uploads."""
    if not data:
        return ""
    text = ""
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        return ""
    return text
