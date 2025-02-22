import os
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using pdfminer.six.
    """
    try:
        text = pdf_extract_text(file_path)
        return text
    except Exception as e:
        return f"Error extracting PDF text: {e}"

def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file using python-docx.
    """
    try:
        document = Document(file_path)
        full_text = [para.text for para in document.paragraphs]
        return "\n".join(full_text)
    except Exception as e:
        return f"Error extracting DOCX text: {e}"

def extract_text_from_txt(file_path):
    """
    Reads text from a TXT file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading TXT file: {e}"

def extract_text_from_file(file_path):
    """
    Determines the file type based on its extension and extracts text.
    Supports PDF, DOCX, and TXT files.
    """
    file_path_lower = file_path.lower()
    if file_path_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path_lower.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path_lower.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return "Unsupported file format."
