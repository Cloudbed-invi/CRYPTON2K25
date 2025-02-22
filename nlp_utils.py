import re
import spacy
from rapidfuzz import fuzz

# Attempt to load a transformer-based spaCy model for improved accuracy.
try:
    nlp = spacy.load('en_core_web_trf')
except Exception as e:
    try:
        nlp = spacy.load('en_core_web_md')
    except Exception as e:
        print("Models 'en_core_web_trf' or 'en_core_web_md' not found. Run: python -m spacy download en_core_web_md")
        raise

WHITELIST_INSTITUTIONS = [
    "Massachusetts Institute of Technology", 
    "Stanford University", 
    "Harvard University"
]
WHITELIST_CERTIFICATIONS = [
    "AWS Certified", "PMP", "Cisco Certified"
]

def clean_candidate_name(name):
    """
    If the candidate name is given as separate letters (e.g., "M A R K"),
    join them together. Otherwise, return the name unchanged.
    """
    tokens = name.split()
    if tokens and all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return name

def is_valid_candidate_name(name):
    """
    Basic check to ensure the extracted name doesn't contain common disqualifiers.
    This avoids names like "MARKETING MANAGER" or organization names.
    """
    disqualifiers = ["university", "college", "manager", "director", "recruiter", "marketing", "engineer"]
    lower = name.lower()
    # Disqualify if any disqualifier is found.
    for word in disqualifiers:
        if word in lower:
            return False
    # Require at least two words for a valid personal name.
    if len(name.split()) < 2:
        return False
    return True

def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "DATE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def validate_email(email):
    pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    return bool(pattern.match(email))

def validate_phone(phone):
    pattern = re.compile(r'\+?\d[\d -]{7,}\d')
    return bool(pattern.match(phone))

def check_against_whitelist(entity, whitelist):
    for item in whitelist:
        if fuzz.token_set_ratio(entity.lower(), item.lower()) > 80:
            return True
    return False

def perform_basic_validation(text):
    email_matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone_matches = re.findall(r'\+?\d[\d -]{7,}\d', text)
    valid_emails = [email for email in email_matches if validate_email(email)]
    valid_phones = [phone for phone in phone_matches if validate_phone(phone)]
    
    entities = extract_entities(text)
    valid_institutions = [org for org in entities.get("ORG", []) if check_against_whitelist(org, WHITELIST_INSTITUTIONS)]
    
    return {
        "emails": valid_emails,
        "phones": valid_phones,
        "valid_institutions": valid_institutions
    }

def clean_candidate_name(name):
    """
    If the candidate name is given as separate letters (e.g., "M A R K"),
    join them together. Otherwise, return the name unchanged.
    """
    tokens = name.split()
    if tokens and all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return name

def is_valid_candidate_name(name):
    """
    Returns False if the extracted name contains disqualifiers or format characters.
    Specifically, if any period ('.') is found in the name (to filter out file formats and org suffixes),
    or if the name includes common organizational terms.
    Additionally, the name must have at least two words and be in proper title case.
    """
    # Reject if there's any period.
    if '.' in name:
        return False

    disqualifiers = [
        "university", "college", "manager", "director",
        "recruiter", "marketing", "engineer", "project", "resume", "file", "cv", "inc", "ltd"
    ]
    lower = name.lower()
    for word in disqualifiers:
        if word in lower:
            return False
    if len(name.split()) < 2:
        return False
    # Reject if the name is all lowercase.
    if name == name.lower():
        return False
    # For names that aren't fully uppercase, enforce that each word starts with an uppercase letter.
    if not name.isupper():
        words = name.split()
        if not all(word[0].isupper() for word in words if word):
            return False
    return True

def extract_candidate_name(text):
    """
    Attempts to extract the candidate's name using a full-document NER approach and filters
    out false positives (like file names or organizational formats) based on the presence
    of periods and disqualifiers. If no valid PERSON entity is found, a fallback heuristic
    examines the first 10 lines.
    """
    # Full-document NER pass using spaCy.
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    for candidate in persons:
        candidate_clean = clean_candidate_name(candidate)
        # Extra check: if candidate contains a period, skip it.
        if '.' in candidate_clean:
            continue
        # Also skip candidates that look like filenames.
        if any(ext in candidate_clean.lower() for ext in ['.pdf', '.doc', '.docx']):
            continue
        if is_valid_candidate_name(candidate_clean):
            return candidate_clean

    # Fallback: Examine the first 10 lines for a potential candidate name.
    lines = text.splitlines()
    job_titles = {"ATTORNEY", "LAWYER", "ENGINEER", "MANAGER", "DIRECTOR", "DEVELOPER", "CONSULTANT", "ANALYST"}
    section_headers = {
        "ACTIVITIES", "ACTIVITIES AND INTERESTS", "EXPERIENCE", "EDUCATION",
        "SKILLS", "CERTIFICATIONS", "SUMMARY", "OBJECTIVE", "PROFILE", "CONTACT", "AWARDS", "PROJECTS", "HOBBIES"
    }
    for line in lines[:10]:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(char.isdigit() for char in line_clean):
            continue
        if line_clean.upper() in section_headers:
            continue
        if any(ext in line_clean.lower() for ext in ['.pdf', '.doc', '.docx']):
            continue
        candidate_clean = clean_candidate_name(line_clean)
        if candidate_clean.upper() in job_titles:
            continue
        if '.' in candidate_clean:
            continue
        if is_valid_candidate_name(candidate_clean):
            return candidate_clean
    return None


    # Fallback: Examine the first 10 lines for a potential candidate name.
    lines = text.splitlines()
    job_titles = {"ATTORNEY", "LAWYER", "ENGINEER", "MANAGER", "DIRECTOR", "DEVELOPER", "CONSULTANT", "ANALYST"}
    section_headers = {
        "ACTIVITIES", "ACTIVITIES AND INTERESTS", "EXPERIENCE", "EDUCATION",
        "SKILLS", "CERTIFICATIONS", "SUMMARY", "OBJECTIVE", "PROFILE", "CONTACT", "AWARDS", "PROJECTS", "HOBBIES"
    }
    for line in lines[:10]:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(char.isdigit() for char in line_clean):
            continue
        if line_clean.upper() in section_headers:
            continue
        if any(ext in line_clean.lower() for ext in ['.pdf', '.doc', '.docx']):
            continue
        candidate_clean = clean_candidate_name(line_clean)
        if candidate_clean.upper() in job_titles:
            continue
        if is_valid_candidate_name(candidate_clean):
            return candidate_clean
    return None

def extract_cleaned_data(text):
    """
    Extracts key fields from the resume text by searching for section headers
    (Skills, Certificates, Universities, Languages) at the beginning of lines.
    Returns 'Not Found' if a section is missing.
    """
    cleaned = re.sub(r'\s+', ' ', text).strip()
    data = {}
    sections = {
        "skills": re.compile(r"(?mi)^skills[:\-]\s*(.+)$"),
        "certificates": re.compile(r"(?mi)^certificates?[:\-]\s*(.+)$"),
        "universities": re.compile(r"(?mi)^universit(?:y|ies)[:\-]\s*(.+)$"),
        "languages": re.compile(r"(?mi)^languages?[:\-]\s*(.+)$")
    }
    for key, pattern in sections.items():
        matches = pattern.findall(cleaned)
        data[key] = " ".join(matches).strip() if matches else "Not Found"
    return data
