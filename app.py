import os
import zipfile
import tempfile
import uuid
import base64
import re
from collections import Counter
from flask import Flask, render_template, request, flash, url_for
from transformers import pipeline
from file_parser import extract_text_from_file
from nlp_utils import (
    extract_entities, 
    perform_basic_validation, 
    extract_candidate_name, 
    extract_cleaned_data
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secure key

# Initialize the summarization pipeline (make sure to install PyTorch)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Global dictionaries to store data
FILE_STORE = {}    # Maps summary_id -> file bytes for PDFs
RAW_TEXT = {}      # Maps summary_id -> extracted text for resumes
SUMMARY_DATA = {}  # Maps summary_id -> summarized text

def compute_score_robust(resume_text, job_title, required_skills, required_languages, min_skills, min_languages, enable_ats, enable_bonus, extra_bonus_keywords, extra_universities):
    """
    Computes a robust score for a resume with a detailed breakdown.
    Components:
      1. Skills + Job Requirement Matching (max = 5 points):
         - Required Skills Matching (max = 4 points; based on the actual number of skills provided)
         - Job Title Matching (max = 1 point)
      2. Languages Matching (if provided; max = 1 point)
      3. Bonus Keywords (if enabled; max = 3 points)
      4. ATS Compatibility (if enabled; max = 2 points)
    
    Returns a tuple: (final_score, breakdown)
    """
    base_text = resume_text.lower() if resume_text else ""
    
    # Component 1: Skills + Job Requirement Matching (max 5)
    required_skills_list = [s.strip().lower() for s in required_skills.split(',') if s.strip()]
    if required_skills_list:
        matched_skills = sum(1 for s in required_skills_list if s in base_text)
        # Use the actual number of skills provided as the denominator.
        denominator = len(required_skills_list)
        skills_score = (matched_skills / denominator) * 4
        skills_score = min(skills_score, 4)
    else:
        skills_score = 0

    # Job Title Matching (max 1)
    job_title_score = 0
    if job_title:
        job_title_lower = job_title.lower().strip()
        if job_title_lower in base_text:
            job_title_score = 1
        else:
            words = job_title_lower.split()
            if words:
                match = sum(1 for word in words if word in base_text)
                job_title_score = min(match / len(words), 1)
    component1_score = skills_score + job_title_score  # Combined max = 5

    # Component 2: Languages Matching (max 1)
    language_score = 0
    if required_languages:
        languages_list = [s.strip().lower() for s in required_languages.split(',') if s.strip()]
        if languages_list:
            matched_languages = sum(1 for lang in languages_list if lang in base_text)
            if min_languages and min_languages.isdigit() and int(min_languages) > 0:
                min_languages_val = int(min_languages)
                language_score = 1 if matched_languages >= min_languages_val else (matched_languages / min_languages_val)
            else:
                language_score = (matched_languages / len(languages_list))
            language_score = language_score * 1
    else:
        language_score = 0

    # Component 3: Bonus Keywords (max 3)
    bonus_score = 0.0
    if enable_bonus == "yes":
        bonus_keywords = {
            'aws certified': 1.0,
            'cisco certified': 1.0,
            'pmp': 1.0,
            'google': 0.5,
            'microsoft': 0.5,
            'oracle': 0.5,
            'ibm': 0.5,
            'certified': 0.3,
            'scrum': 0.3,
        }
        if extra_bonus_keywords:
            for kw in extra_bonus_keywords.split(','):
                kw = kw.strip().lower()
                if kw:
                    bonus_keywords[kw] = 0.5
        if extra_universities:
            for uni in extra_universities.split(','):
                uni = uni.strip().lower()
                if uni:
                    bonus_keywords[uni] = 0.5
        for keyword, points in bonus_keywords.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', base_text, re.IGNORECASE):
                bonus_score += points
        bonus_score = min(bonus_score, 3)
    else:
        bonus_score = 0

    # Component 4: ATS Compatibility (max 2)
    ats_score = 0
    if enable_ats == "yes":
        ats_keywords = ["experience", "education", "skills", "certification", "projects", "summary", "objective", "profile"]
        ats_count = sum(1 for word in ats_keywords if word in base_text)
        ats_score = min((ats_count / len(ats_keywords)) * 2, 2)
    else:
        ats_score = 0

    raw_total = component1_score + language_score + bonus_score + ats_score
    
    total_max = 5  # Component 1 is always enabled.
    if required_languages:
        total_max += 1
    if enable_bonus == "yes":
        total_max += 3
    if enable_ats == "yes":
        total_max += 2

    final_score = (raw_total / total_max) * 10 if total_max > 0 else 0

    breakdown = {
        "required_skills": round(skills_score, 2),
        "job_title": round(job_title_score, 2),
        "skills_job": round(component1_score, 2),
        "languages": round(language_score, 2),
        "bonus": round(bonus_score, 2),
        "ats": round(ats_score, 2),
        "raw_total": round(raw_total, 2),
        "total_max": total_max
    }
    return round(final_score, 2), breakdown

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash("No file part in the request.")
        return render_template('index.html')
    
    files = request.files.getlist('files')
    if not files:
        flash("No files uploaded.")
        return render_template('index.html')
    
    job_title = request.form.get('job_title')
    required_skills = request.form.get('skills')
    required_languages = request.form.get('languages', '')
    min_skills = request.form.get('min_skills', '')
    min_languages = request.form.get('min_languages', '')
    enable_ats = request.form.get('enable_ats', 'no')
    enable_bonus = request.form.get('enable_bonus', 'no')
    extra_bonus_keywords = request.form.get('extra_bonus_keywords', '')
    extra_universities = request.form.get('extra_universities', '')
    
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
                                    file_bytes = zip_ref.read(name)
                                    tmp_file.write(file_bytes)
                                    tmp_file.close()
                                    extracted_text = extract_text_from_file(tmp_file.name)
                                    entities = extract_entities(extracted_text)
                                    validation = perform_basic_validation(extracted_text)
                                    candidate_name = extract_candidate_name(extracted_text)
                                    score, breakdown = compute_score_robust(
                                        extracted_text,
                                        job_title,
                                        required_skills,
                                        required_languages,
                                        min_skills,
                                        min_languages,
                                        enable_ats,
                                        enable_bonus,
                                        extra_bonus_keywords,
                                        extra_universities
                                    )
                                    summary_id = str(uuid.uuid4())
                                    ext_lower = os.path.splitext(name)[1].lower()
                                    if ext_lower == '.pdf':
                                        FILE_STORE[summary_id] = {
                                            'type': 'pdf',
                                            'data': file_bytes
                                        }
                                        RAW_TEXT[summary_id] = extracted_text
                                    else:
                                        RAW_TEXT[summary_id] = extracted_text
                                    results.append({
                                        'file': name,
                                        'text': extracted_text,
                                        'candidate_name': candidate_name,
                                        'entities': entities,
                                        'validation': validation,
                                        'score': score,
                                        'score_breakdown': breakdown,
                                        'summary_id': summary_id
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
                    score, breakdown = compute_score_robust(
                        extracted_text,
                        job_title,
                        required_skills,
                        required_languages,
                        min_skills,
                        min_languages,
                        enable_ats,
                        enable_bonus,
                        extra_bonus_keywords,
                        extra_universities
                    )
                    summary_id = str(uuid.uuid4())
                    ext_lower = ext.lower()
                    if ext_lower == '.pdf':
                        with open(tmp_file.name, 'rb') as f:
                            file_bytes = f.read()
                        FILE_STORE[summary_id] = {
                            'type': 'pdf',
                            'data': file_bytes
                        }
                        RAW_TEXT[summary_id] = extracted_text
                    else:
                        RAW_TEXT[summary_id] = extracted_text
                    results.append({
                        'file': file.filename,
                        'text': extracted_text,
                        'candidate_name': candidate_name,
                        'entities': entities,
                        'validation': validation,
                        'score': score,
                        'score_breakdown': breakdown,
                        'summary_id': summary_id
                    })
                finally:
                    os.remove(tmp_file.name)
        except Exception as e:
            results.append({'file': file.filename, 'text': f"Error processing file: {str(e)}"})
    
    results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    total_candidates = len(results)
    average_score = round(sum(r.get('score', 0) for r in results) / total_candidates, 2) if total_candidates > 0 else 0
    highest_score = max(r.get('score', 0) for r in results) if total_candidates > 0 else 0
    lowest_score = min(r.get('score', 0) for r in results) if total_candidates > 0 else 0

    return render_template('result.html', results=results, job_title=job_title, required_skills=required_skills,
                           required_languages=required_languages, total_candidates=total_candidates,
                           average_score=average_score, highest_score=highest_score, lowest_score=lowest_score)

@app.route('/summary')
def summary():
    sid = request.args.get('id')
    if sid:
        if sid in SUMMARY_DATA:
            summary_text = SUMMARY_DATA[sid]
        else:
            raw_text = RAW_TEXT.get(sid, None)
            if raw_text:
                try:
                    summary = summarizer(raw_text, max_length=250, min_length=50, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    SUMMARY_DATA[sid] = summary_text
                except Exception as e:
                    summary_text = "Summary not available."
            else:
                summary_text = "No text available for summarization."
        return render_template('summary.html', data=summary_text)
    else:
        return "No summary ID provided", 404

@app.route('/full')
def full():
    sid = request.args.get('id')
    if sid:
        file_entry = FILE_STORE.get(sid)
        if file_entry and file_entry['type'] == 'pdf':
            pdf_data = file_entry['data']
            b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            return render_template('full.html', data=b64_pdf, is_pdf=True)
        else:
            raw_text = RAW_TEXT.get(sid, None)
            if raw_text:
                return render_template('full.html', data=raw_text, is_pdf=False)
            else:
                return "No resume text available.", 404
    else:
        return "No resume ID provided", 404

@app.route('/dashboard')
def dashboard():
    # Aggregate all resume texts.
    all_text = " ".join(RAW_TEXT.values())
    
    # Buzzwords frequency
    buzzwords = ['aws', 'certified', 'cisco', 'pmp', 'google', 'microsoft', 'oracle', 'ibm', 'scrum', 'python', 'java', 'sql']
    buzzword_counter = Counter()
    for word in buzzwords:
        matches = re.findall(r'\b' + re.escape(word) + r'\b', all_text, re.IGNORECASE)
        buzzword_counter[word] = len(matches)
    
    # Extract skills from resumes (lines starting with "skills:")
    skills_counter = Counter()
    for text in RAW_TEXT.values():
        matches = re.findall(r'(?mi)^skills[:\-]\s*(.+)$', text)
        for match in matches:
            for skill in match.split(','):
                skill = skill.strip().lower()
                if skill:
                    skills_counter[skill] += 1
                    
    # Programming languages (predefined list)
    prog_langs = ['python', 'java', 'c++', 'javascript', 'ruby', 'go', 'php', 'c#']
    prog_lang_counter = Counter()
    for lang in prog_langs:
        matches = re.findall(r'\b' + re.escape(lang) + r'\b', all_text, re.IGNORECASE)
        prog_lang_counter[lang] = len(matches)
    
    # Soft skills (predefined list)
    soft_skills = ['communication', 'teamwork', 'problem-solving', 'adaptability', 'leadership', 'creativity']
    soft_skill_counter = Counter()
    for skill in soft_skills:
        matches = re.findall(r'\b' + re.escape(skill) + r'\b', all_text, re.IGNORECASE)
        soft_skill_counter[skill] = len(matches)
    
    top_buzzwords = buzzword_counter.most_common(10)
    top_skills = skills_counter.most_common(10)
    prog_lang_counts = [prog_lang_counter[lang] for lang in prog_langs]
    soft_skill_counts = [soft_skill_counter[skill] for skill in soft_skills]
    
    buzz_labels = [item[0] for item in top_buzzwords]
    buzz_data = [item[1] for item in top_buzzwords]
    
    skill_labels = [item[0] for item in top_skills]
    skill_data = [item[1] for item in top_skills]
    
    return render_template('dashboard.html', buzz_labels=buzz_labels, buzz_data=buzz_data,
                           skill_labels=skill_labels, skill_data=skill_data,
                           prog_lang_counts=prog_lang_counts, soft_skill_counts=soft_skill_counts)

if __name__ == '__main__':
    FILE_STORE = {}
    RAW_TEXT = {}
    SUMMARY_DATA = {}
    app.run(debug=True)
