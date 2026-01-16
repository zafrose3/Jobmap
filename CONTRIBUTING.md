# Contributing to JobMap

Thank you for your interest in contributing to **JobMap**! We want to make this project a great resource for developers to learn about AI, Data Engineering, and Full-stack development.

## 🌟 How to Contribute

### 1. Reporting Bugs
- Check the Issues tab to see if the bug has already been reported.
- Open a new Issue describing the bug, steps to reproduce, and expected behavior.

### 2. Suggesting Features
- We love new ideas! Open an Issue with the "Enhancement" label to discuss your proposal.
- Check our `task.md` or `implementation_plan.md` to see what's currently planned.

### 3. Submitting Code
1. **Fork** the repository.
2. **Clone** your fork locally.
3. Create a new branch: `git checkout -b feature/amazing-feature`.
4. Make your changes and commit: `git commit -m "Add amazing feature"`.
5. Push to your branch: `git push origin feature/amazing-feature`.
6. Open a **Pull Request**.

## 💻 Development Guidelines

### Backend (Python/FastAPI)
- Follow **PEP 8** style guidelines.
- Use **SQLModel** for all database interactions.
- Ensure all new endpoints have type hints.
- Run tests (if available) before submitting.

### Frontend (Next.js/React)
- Use **functional components** and Hooks.
- Style with **Tailwind CSS** utilities (avoid custom CSS files where possible).
- Ensure the UI is responsive on mobile devices.

### Directory Structure
```
JobMap/
├── backend/            # FastAPI app & logic
│   ├── data/           # Raw data files (CSVs)
│   ├── main.py         # App entry point
│   ├── models.py       # DB Schemas
│   └── analysis.py     # ML/NLP logic
└── frontend/           # Next.js app
    ├── src/app/        # Pages
    └── src/components/ # Reusable UI components
```

## 🎓 Beginner Friendly
This is a great project for first-time contributors. Look for issues labeled `good first issue`!

## 📜 Code of Conduct
Please be respectful and kind. Harassment of any kind will not be tolerated.

Happy Coding! 🚀
