# CI/CD Guide — PQNK AgriChat Platform

## Overview

This guide covers:
1. **Local development checks** (linting + unit tests before every commit)
2. **GitHub Actions CI pipeline** (automated on every push/PR)
3. **Pre-deployment checklist**

---

## 1. Local Development Checks

### Backend (Python / Flask)

```bash
# Install dev dependencies (one-time)
pip install flake8 pytest pytest-flask

# Lint — PEP 8 style check
cd agri_ui
flake8 app.py pipeline_service.py rag_demo.py drive_service.py --max-line-length=120 --ignore=E501,W503

# Run tests (add test files under agri_ui/tests/)
pytest tests/ -v

# Run the Flask backend locally
cd agri_ui
python app.py
```

### Frontend (React / Vite)

```bash
# Install dependencies (one-time)
cd frontend-react/PQNK_Frontend
npm install

# Lint — ESLint
npm run lint

# Build check — ensures no compile errors
npm run build

# Dev server
npm run dev
```

---

## 2. GitHub Actions CI Pipeline

Create this file in your repo: `.github/workflows/ci.yml`

```yaml
name: PQNK CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:

  # ── Backend checks ───────────────────────────────────────────────────
  backend:
    name: Backend — Lint & Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest pytest-flask flask flask-login flask-bcrypt flask-sqlalchemy flask-cors python-dotenv

      - name: Lint with flake8
        run: |
          cd agri_ui
          flake8 app.py pipeline_service.py drive_service.py \
            --max-line-length=120 \
            --ignore=E501,W503,E402

      - name: Run backend tests
        run: |
          cd agri_ui
          pytest tests/ -v --tb=short
        env:
          SECRET_KEY: test-ci-key
          FLASK_ENV: testing

  # ── Frontend checks ──────────────────────────────────────────────────
  frontend:
    name: Frontend — Lint & Build
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Node.js 20
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: frontend-react/PQNK_Frontend/package-lock.json

      - name: Install frontend dependencies
        run: |
          cd frontend-react/PQNK_Frontend
          npm ci

      - name: Lint frontend (ESLint)
        run: |
          cd frontend-react/PQNK_Frontend
          npm run lint

      - name: Build production bundle
        run: |
          cd frontend-react/PQNK_Frontend
          npm run build
        env:
          VITE_API_BASE_URL: http://localhost:5000
```

> **Important:** The CI job will pass without tests if the `agri_ui/tests/` directory doesn't exist yet. Create it with at least a placeholder file:
>
> ```python
> # agri_ui/tests/test_health.py
> def test_placeholder():
>     """Placeholder test — passes CI until real tests are added."""
>     assert True
> ```

---

## 3. Writing Your First Backend Tests

Create `agri_ui/tests/conftest.py`:

```python
import pytest
from app import app as flask_app

@pytest.fixture
def app():
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["WTF_CSRF_ENABLED"] = False
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()
```

Create `agri_ui/tests/test_auth.py`:

```python
def test_signup_missing_fields(client):
    """Signup without required fields returns 400."""
    res = client.post("/api/signup", json={})
    assert res.status_code == 400

def test_signup_weak_password(client):
    """Signup with a weak password is rejected."""
    res = client.post("/api/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "password",   # no uppercase, no special char
    })
    assert res.status_code == 400
    data = res.get_json()
    assert not data["success"]

def test_login_invalid_credentials(client):
    """Login with wrong credentials returns 401."""
    res = client.post("/api/login", json={
        "email": "nobody@test.com",
        "password": "SomePass1!",
    })
    assert res.status_code == 401
```

---

## 4. Environment Variables Checklist

Before any CI run or deployment, ensure these vars are set in GitHub Secrets (Settings → Secrets → Actions):

| Secret Name        | Description                               |
|--------------------|-------------------------------------------|
| `SECRET_KEY`       | Flask session secret (32+ random chars)  |
| `OPENAI_API_KEY`   | OpenAI API key for embeddings            |
| `QDRANT_URL`       | Qdrant Cloud URL                          |
| `QDRANT_API_KEY`   | Qdrant API key                            |
| `MAIL_USERNAME`    | Gmail address for OTP sending            |
| `MAIL_PASSWORD`    | Gmail App Password (16 chars)            |
| `DATABASE_URL`     | PostgreSQL connection string             |

---

## 5. Branch Strategy

```
main        ← production-ready code only
develop     ← active development
feature/*   ← individual features / bug fixes
```

- PRs to `main` require at least 1 approval + all CI checks passing
- PRs to `develop` require CI checks passing

---

## 6. Pre-Deployment Checklist

Before pushing to production:

- [ ] `SECRET_KEY` set to a proper random value in `.env`
- [ ] All CI checks pass on the `main` branch
- [ ] `npm run build` produces no errors
- [ ] Database migrations applied (`flask db upgrade` if using Flask-Migrate)
- [ ] `CORS` origins updated to the production frontend URL
- [ ] Google Drive credentials file deployed and path configured
- [ ] Qdrant collection exists and has vectors
- [ ] MAIL_PASSWORD is a valid Gmail App Password
