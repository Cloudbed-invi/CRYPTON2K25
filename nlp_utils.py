import re
import spacy
from fuzzywuzzy import fuzz

try:
    nlp = spacy.load('en_core_web_sm')
except IOError:
    print("Model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise

WHITELIST_INSTITUTIONS = [
    "Massachusetts Institute of Technology", 
    "Stanford University", 
    "Harvard University"
]
WHITELIST_CERTIFICATIONS = [
    "AWS Certified", "PMP", "Cisco Certified"
]

def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "DATE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    pattern = r'\+?\d[\d -]{7,}\d'
    return bool(re.match(pattern, phone))

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
    valid_institutions = []
    for org in entities.get("ORG", []):
        if check_against_whitelist(org, WHITELIST_INSTITUTIONS):
            valid_institutions.append(org)
    
    return {
        "emails": valid_emails,
        "phones": valid_phones,
        "valid_institutions": valid_institutions
    }

def extract_candidate_name(text):
    lines = text.splitlines()
    candidate_name = None
    job_titles = {"ATTORNEY", "LAWYER", "ENGINEER", "MANAGER", "DIRECTOR", "DEVELOPER", "CONSULTANT", "ANALYST"}
    section_headers = {"ACTIVITIES", "ACTIVITIES AND INTERESTS", "EXPERIENCE", "EDUCATION", 
                       "SKILLS", "CERTIFICATIONS", "SUMMARY", "OBJECTIVE", "PROFILE", "CONTACT", "AWARDS", "PROJECTS", "HOBBIES"}
    
    for line in lines[:10]:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(char.isdigit() for char in line_clean):
            continue
        if line_clean.upper() in section_headers:
            continue
        
        doc_line = nlp(line_clean)
        persons = [ent.text for ent in doc_line.ents if ent.label_ == "PERSON"]
        if persons:
            return persons[0]
        
        if (line_clean.isupper() or line_clean.istitle()) and len(line_clean.split()) >= 2:
            if line_clean.upper() not in job_titles:
                candidate_name = line_clean
                break
    return candidate_name

def extract_cleaned_data(text):
    """
    Extracts key fields from the resume text. This function searches for section headers
    (Skills, Certificates, Universities, Languages) at the beginning of lines (multiline mode).
    If not found, it returns "Not Found" for that field.
    """
    cleaned = re.sub(r'\s+', ' ', text).strip()
    data = {}
    sections = {
        "skills": r"(?mi)^skills[:\-]\s*(.+)$",
        "certificates": r"(?mi)^certificates?[:\-]\s*(.+)$",
        "universities": r"(?mi)^universit(?:y|ies)[:\-]\s*(.+)$",
        "languages": r"(?mi)^languages?[:\-]\s*(.+)$"
    }
    for key, pattern in sections.items():
        matches = re.findall(pattern, cleaned)
        if matches:
            data[key] = " ".join(matches).strip()
        else:
            data[key] = "Not Found"
    return data
