import os
import zipfile
import tempfile
from flask import Flask, render_template, request, flash, url_for
from file_parser import extract_text_from_file

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Ensure the form field is named "files"
    if 'files' not in request.files:
        flash("No file part in the request.")
        return render_template('index.html')

    files = request.files.getlist('files')
    if not files:
        flash("No files uploaded.")
        return render_template('index.html')

    results = []

    for file in files:
        if file.filename == '':
            continue  # Skip files with empty filenames

        filename = file.filename.lower()
        try:
            # If the file is a zip archive, extract its contents
            if filename.endswith('.zip'):
                # Save the zip file to a temporary file
                tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                try:
                    file.save(tmp_zip.name)
                    tmp_zip.close()  # Close before using it
                    with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
                        for name in zip_ref.namelist():
                            # Process only supported file types from the zip
                            if name.endswith(('.pdf', '.docx', '.txt')):
                                # Save each entry to a temporary file
                                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
                                try:
                                    tmp_file.write(zip_ref.read(name))
                                    tmp_file.close()  # Close to release the file handle
                                    extracted_text = extract_text_from_file(tmp_file.name)
                                    results.append({'file': name, 'text': extracted_text})
                                finally:
                                    os.remove(tmp_file.name)
                finally:
                    os.remove(tmp_zip.name)
            else:
                # Process an individual resume file
                ext = os.path.splitext(file.filename)[1]
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                try:
                    file.save(tmp_file.name)
                    tmp_file.close()  # Explicitly close the file so it can be read
                    extracted_text = extract_text_from_file(tmp_file.name)
                    results.append({'file': file.filename, 'text': extracted_text})
                finally:
                    os.remove(tmp_file.name)
        except Exception as e:
            results.append({'file': file.filename, 'text': f"Error processing file: {str(e)}"})

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
