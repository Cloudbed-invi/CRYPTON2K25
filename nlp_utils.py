import re
import spacy
from fuzzywuzzy import fuzz

# Load spaCy model (ensure you have run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load('en_core_web_sm')
except IOError:
    print("Model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise

# Whitelists for basic validation
WHITELIST_INSTITUTIONS = [
    "Massachusetts Institute of Technology", 
    "Stanford University", 
    "Harvard University"
]
WHITELIST_CERTIFICATIONS = [
    "AWS Certified", "PMP", "Cisco Certified"
]

def extract_entities(text):
    """
    Extract named entities (like PERSON, ORG, DATE) using spaCy.
    """
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "DATE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def validate_email(email):
    """
    Validate email format using a regex.
    """
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    """
    Validate a basic phone number pattern.
    """
    pattern = r'\+?\d[\d -]{7,}\d'
    return bool(re.match(pattern, phone))

def check_against_whitelist(entity, whitelist):
    """
    Use fuzzy matching to check if an entity is similar to an entry in the whitelist.
    """
    for item in whitelist:
        if fuzz.token_set_ratio(entity.lower(), item.lower()) > 80:
            return True
    return False

def perform_basic_validation(text):
    """
    Performs basic validations on text:
      - Validates emails and phone numbers.
      - Checks if extracted organization names match a whitelist.
    """
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
    """
    Heuristically extracts the candidate's name from the resume text.
    It examines the first few lines and:
      - Skips empty lines or lines containing digits.
      - Ignores common section headers.
      - Uses spaCy's NER on each line; if a PERSON entity is found, returns it.
      - Otherwise, if the line is in uppercase or title case and contains at least two words,
        returns it as the candidate name.
    """
    lines = text.splitlines()
    candidate_name = None
    # Define known job titles and common section headers to ignore
    job_titles = {"ATTORNEY", "LAWYER", "ENGINEER", "MANAGER", "DIRECTOR", "DEVELOPER", "CONSULTANT", "ANALYST"}
    section_headers = {"ACTIVITIES", "ACTIVITIES AND INTERESTS", "EXPERIENCE", "EDUCATION", 
                       "SKILLS", "CERTIFICATIONS", "SUMMARY", "OBJECTIVE", "PROFILE", "CONTACT", "AWARDS", "PROJECTS", "HOBBIES"}
    
    # Examine only the first 10 lines of the resume
    for line in lines[:10]:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Skip lines that contain digits (often dates or numbers)
        if any(char.isdigit() for char in line_clean):
            continue
        # Skip lines that exactly match known section headers
        if line_clean.upper() in section_headers:
            continue
        
        # Use spaCy's NER to check for a PERSON entity in the line
        doc_line = nlp(line_clean)
        persons = [ent.text for ent in doc_line.ents if ent.label_ == "PERSON"]
        if persons:
            return persons[0]
        
        # Otherwise, if the line is all uppercase or title case and not a known job title, use it if it has at least 2 words.
        if (line_clean.isupper() or line_clean.istitle()) and len(line_clean.split()) >= 2:
            if line_clean.upper() not in job_titles:
                candidate_name = line_clean
                break
    return candidate_name
