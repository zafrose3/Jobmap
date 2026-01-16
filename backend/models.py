from typing import Optional, List, TYPE_CHECKING
from sqlmodel import Field, SQLModel, Relationship
from datetime import datetime

# Forward references for type checking
if TYPE_CHECKING:
    from .models import Job, Skill

class JobSkillLink(SQLModel, table=True):
    job_id: Optional[int] = Field(default=None, foreign_key="job.id", primary_key=True)
    skill_id: Optional[int] = Field(default=None, foreign_key="skill.id", primary_key=True)

class JobBase(SQLModel):
    title: str
    company: str
    location: Optional[str] = None
    description: str
    source: str
    posted_date: Optional[datetime] = None
    url: Optional[str] = None

class Job(JobBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    skills: List["Skill"] = Relationship(back_populates="jobs", link_model=JobSkillLink)

class SkillBase(SQLModel):
    name: str = Field(index=True, unique=True)
    category: Optional[str] = None

class Skill(SkillBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    jobs: List[Job] = Relationship(back_populates="skills", link_model=JobSkillLink)
