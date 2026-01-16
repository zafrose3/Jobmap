import csv
import os
from sqlmodel import Session, select
from .database import engine, create_db_and_tables
from .models import Job, Skill, JobSkillLink
from .analysis import extract_skills
from datetime import datetime

DATA_FILE = "backend/data/jobs.csv"

def ingest_jobs():
    print("Creating DB and tables...")
    create_db_and_tables()
    
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found.")
        return

    print(f"Reading from {DATA_FILE}...")
    
    with Session(engine) as session:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if job already exists (by URL or specific fields)
                # specific lookup could be expensive, here we just insert for MVP
                
                job = Job(
                    title=row["title"],
                    company=row["company"],
                    location=row["location"],
                    description=row["description"],
                    source=row["source"],
                    url=row["url"],
                    posted_date=datetime.now() # Mock date for now
                )
                session.add(job)
                session.commit() # Commit to get ID
                session.refresh(job) # Refresh to get ID
                
                # Extract Skills
                raw_skills = extract_skills(row["description"])
                
                for skill_name in raw_skills:
                    # Check if skill exists
                    statement = select(Skill).where(Skill.name == skill_name)
                    skill = session.exec(statement).first()
                    
                    if not skill:
                        skill = Skill(name=skill_name, category="Tech") # Default cat
                        session.add(skill)
                        session.commit()
                        session.refresh(skill)
                    
                    # Link Job and Skill
                    # Check if link exists? (Ideally yes, but for new job/skill just add)
                    link = JobSkillLink(job_id=job.id, skill_id=skill.id)
                    session.add(link)
                
                print(f"Ingested job: {job.title} with skills: {raw_skills}")
            
            session.commit()
    print("Ingestion complete.")

if __name__ == "__main__":
    # Since we are running as a module usually, specific path handling might be needed
    # if run directly: python -m backend.ingest_data
    ingest_jobs()
