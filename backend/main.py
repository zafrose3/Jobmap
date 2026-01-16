from fastapi import FastAPI, Depends
from sqlmodel import Session, select
from contextlib import asynccontextmanager
from .database import create_db_and_tables, get_session
from .models import Job, Skill

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title="JobMap API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to JobMap API"}

@app.get("/jobs", response_model=list[Job])
def read_jobs(skip: int = 0, limit: int = 100, session: Session = Depends(get_session)):
    jobs = session.exec(select(Job).offset(skip).limit(limit)).all()
    return jobs
