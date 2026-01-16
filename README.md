# JobMap 🗺️

**JobMap** is an AI-powered career intelligence platform that analyzes recent job postings to identify hiring trends, in-demand skills, and evolving job roles. Designed to be open-source and beginner-friendly, it helps job seekers navigate the market with data-driven insights.

## 🚀 Features

- **Trend Analysis**: Visualize top in-demand technical skills over time.
- **Skill Extraction**: Uses NLP (spaCy) to parse job descriptions and extract key technologies.
- **Job Explorer**: Browse recent job postings from aggregated sources.
- **Interactive Dashboard**: Built with Next.js and Recharts for clear data visualization.
- **API First**: Robust FastAPI backend serving career data.

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, SQLModel, PostgreSQL/SQLite.
- **AI/ML**: spaCy (NLP), scikit-learn (Analysis).
- **Frontend**: Next.js (React), Tailwind CSS, Recharts.
- **Data**: Kaggle Datasets, Public APIs.

## 🏁 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend Setup

```bash
cd backend
python -m venv .venv
# Windows
.\.venv\Scripts\Activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run Data Ingestion (Seeds the DB with dummy data)
python -m backend.ingest_data

# Start Server
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000` (Docs at `/docs`).

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to view the dashboard.

## 🔮 Roadmap

- [x] Basic Dashboard & Skill Extraction
- [ ] Chatbot Interface for Career Advice
- [ ] Real-time Data Piping via Celery
- [ ] User Authentication

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## 📄 License

MIT
