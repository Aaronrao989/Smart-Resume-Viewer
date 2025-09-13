from typing import Tuple
import fitz  # PyMuPDF
import pdfplumber
from .utils import clean_text

def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """Return (text, page_count). Try PyMuPDF first then fallback to pdfplumber."""
    text = ""
    page_count = 0
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        for p in doc:
            t = p.get_text("text")
            if t:
                text += "\n" + t
        doc.close()
        text = clean_text(text)
        if text:
            return text, page_count
    except Exception:
        pass

    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                text += "\n" + t
        text = clean_text(text)
        return text, page_count
    except Exception as e:
        return "", 0
