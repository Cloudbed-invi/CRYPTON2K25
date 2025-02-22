from file_parser import extract_text_from_file

pdf_path = "Resumes/Attorney resume.pdf"
docx_path = "Resumes/Industry manager resume.docx"
txt_path = "Resumes/Project management resume.txt"

print("PDF Extraction:")
print(extract_text_from_file(pdf_path))
print("\nDOCX Extraction:")
print(extract_text_from_file(docx_path))
print("\nTXT Extraction:")
print(extract_text_from_file(txt_path))
