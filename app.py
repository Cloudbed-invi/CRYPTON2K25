import os
import zipfile
import tempfile
from flask import Flask, render_template, request, flash, url_for
from file_parser import extract_text_from_file
from nlp_utils import extract_entities, perform_basic_validation, extract_candidate_name

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this for production

def compute_score(resume_text, required_skills):
    """
    A simple scoring function:
      - Splits the comma-separated required skills.
      - For each required skill, check if it exists (case-insensitive) in the resume text.
      - Returns a percentage match.
    """
    skills = [skill.strip().lower() for skill in required_skills.split(',') if skill.strip()]
    if not skills:
        return 0
    match_count = 0
    resume_text_lower = resume_text.lower()
    for skill in skills:
        if skill in resume_text_lower:
            match_count += 1
    score_percent = (match_count / len(skills)) * 100
    return round(score_percent, 2)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Verify that the form includes the necessary fields.
    if 'files' not in request.files:
        flash("No file part in the request.")
        return render_template('index.html')
    
    files = request.files.getlist('files')
    if not files:
        flash("No files uploaded.")
        return render_template('index.html')
    
    # Read job requirements from the form.
    job_title = request.form.get('job_title')
    required_skills = request.form.get('skills')

    results = []
    for file in files:
        if file.filename == '':
            continue

        filename = file.filename.lower()
        try:
            if filename.endswith('.zip'):
                tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                try:
                    file.save(tmp_zip.name)
                    tmp_zip.close()
                    with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
                        for name in zip_ref.namelist():
                            if name.endswith(('.pdf', '.docx', '.txt')):
                                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
                                try:
                                    tmp_file.write(zip_ref.read(name))
                                    tmp_file.close()
                                    extracted_text = extract_text_from_file(tmp_file.name)
                                    entities = extract_entities(extracted_text)
                                    validation = perform_basic_validation(extracted_text)
                                    candidate_name = extract_candidate_name(extracted_text)
                                    score = compute_score(extracted_text, required_skills)
                                    results.append({
                                        'file': name,
                                        'text': extracted_text,
                                        'candidate_name': candidate_name,
                                        'entities': entities,
                                        'validation': validation,
                                        'score': score
                                    })
                                finally:
                                    os.remove(tmp_file.name)
                finally:
                    os.remove(tmp_zip.name)
            else:
                ext = os.path.splitext(file.filename)[1]
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                try:
                    file.save(tmp_file.name)
                    tmp_file.close()
                    extracted_text = extract_text_from_file(tmp_file.name)
                    entities = extract_entities(extracted_text)
                    validation = perform_basic_validation(extracted_text)
                    candidate_name = extract_candidate_name(extracted_text)
                    score = compute_score(extracted_text, required_skills)
                    results.append({
                        'file': file.filename,
                        'text': extracted_text,
                        'candidate_name': candidate_name,
                        'entities': entities,
                        'validation': validation,
                        'score': score
                    })
                finally:
                    os.remove(tmp_file.name)
        except Exception as e:
            results.append({'file': file.filename, 'text': f"Error processing file: {str(e)}"})
    
    return render_template('result.html', results=results, job_title=job_title, required_skills=required_skills)

if __name__ == '__main__':
    app.run(debug=True)
