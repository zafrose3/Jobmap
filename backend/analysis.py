import spacy
from collections import Counter
import re

# Load spaCy model (ensure to run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

COMMON_SKILLS = {
    "python", "java", "javascript", "react", "node.js", "sql", "aws", "docker", 
    "kubernetes", "machine learning", "data analysis", "fastapi", "django", "flask",
    "postgresql", "mongodb", "git", "ci/cd", "agile", "scrum", "communication", 
    "teamwork", "problem solving", "linux", "bash", "html", "css", "typescript"
}

def extract_skills(text: str) -> list[str]:
    """
    Extract skills from job description using spaCy and simple keyword matching.
    """
    doc = nlp(text.lower())
    found_skills = set()

    # 1. Simple Keyword Matching (Fast & Effective for tech skills)
    for token in doc:
        if token.text in COMMON_SKILLS:
            found_skills.add(token.text)
    
    # 2. Phrase Matching (for multi-word skills)
    # This is a naive implementation; for production, use spacy `PhraseMatcher`
    text_lower = text.lower()
    for skill in COMMON_SKILLS:
        if " " in skill and skill in text_lower:
            found_skills.add(skill)

    # 3. NER for additional entities (optional, can extract ORG, etc.)
    # for ent in doc.ents:
    #    if ent.label_ == "ORG":
    #        print(ent.text)

    return list(found_skills)

def analyze_trends(jobs_data: list[dict]) -> dict:
    """
    Aggregate skills from a list of jobs to find top trends.
    """
    all_skills = []
    for job in jobs_data:
        all_skills.extend(job.get("extracted_skills", []))
    
    counter = Counter(all_skills)
    return dict(counter.most_common(10))
