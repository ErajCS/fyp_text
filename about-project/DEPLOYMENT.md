# 🌿 PQNK Knowledge Intelligence System — Free Deployment Guide

A step-by-step guide to deploy this project **entirely for free**.

---

## Architecture Overview

```
Users → React Frontend (Vercel) → Flask REST API (Railway) → PostgreSQL (Railway)
                                    ↓
                              Qdrant Cloud (free tier)
                              OpenAI API (paid API key required)
```

---

## Prerequisites

Before starting, make sure you have:
- ✅ A [GitHub](https://github.com) account
- ✅ An [OpenAI](https://platform.openai.com) API key
- ✅ A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier)
- ✅ A [Gmail](https://gmail.com) account with an [App Password](https://myaccount.google.com/apppasswords) configured

---

## Step 1 — Push Code to GitHub

```bash
cd c:\Users\smwaj\fyp_text
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pqnk-system.git
git push -u origin main
```

> **Important:** Ensure `.env` is in `.gitignore` (it already is). Never push secrets to GitHub.

---

## Step 2 — Deploy Backend + Database on Railway (Free)

[Railway](https://railway.app) gives you $5/month free credit — enough for a small Flask + PostgreSQL app.

### 2a. Create a Railway account
1. Go to [railway.app](https://railway.app) → **Login with GitHub**

### 2b. Deploy PostgreSQL
1. New Project → **Add Database** → **PostgreSQL**
2. Click the PostgreSQL service → **Variables** tab → copy `DATABASE_URL`

### 2c. Deploy Flask Backend
1. In the same project → **New Service** → **GitHub Repo** → select your repo
2. Set the root directory: `agri_ui`  
3. Set start command:
   ```
   gunicorn app:app
   ```
4. Add environment variables (click **Variables**):

| Variable | Value |
|---|---|
| `DATABASE_URL` | Paste from step 2b |
| `OPENAI_API_KEY` | Your OpenAI key |
| `QDRANT_URL` | Your Qdrant cluster URL |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `MAIL_USERNAME` | Your Gmail address |
| `MAIL_PASSWORD` | Your Gmail App Password |
| `SECRET_KEY` | Any random 32-char string |

5. Railway will automatically detect Python and install `requirements.txt`
6. Once deployed, copy your backend URL (e.g. `https://pqnk-backend.up.railway.app`)

### 2d. Install Gunicorn (if not already in requirements)
```bash
pip install gunicorn
pip freeze > agri_ui/requirements.txt
git add agri_ui/requirements.txt && git commit -m "Add gunicorn" && git push
```

---

## Step 3 — Update Vite Proxy for Production

In `frontend-react/PQNK_Frontend/vite.config.js`, the proxy target needs to point to your Railway URL **during build**. Instead, for production, the React app will call the API using an environment variable.

Edit `vite.config.js`:
```js
server: {
  proxy: {
    "/api": {
      target: process.env.VITE_API_URL || "http://localhost:5000",
      changeOrigin: true,
    },
    "/get_response": { target: process.env.VITE_API_URL || "http://localhost:5000", changeOrigin: true },
    "/generate_audio": { target: process.env.VITE_API_URL || "http://localhost:5000", changeOrigin: true },
  }
}
```

---

## Step 4 — Deploy React Frontend on Vercel (Free)

[Vercel](https://vercel.com) deploys React/Vite apps for free with unlimited bandwidth.

1. Go to [vercel.com](https://vercel.com) → **Login with GitHub**
2. **New Project** → Import your GitHub repo
3. Set **Root Directory** to: `frontend-react/PQNK_Frontend`
4. Framework preset: **Vite**
5. Add Environment Variable:
   - `VITE_API_URL` = `https://pqnk-backend.up.railway.app` (your Railway URL)
6. Click **Deploy** — Vercel builds and hosts it automatically

---

## Step 5 — Allow Cross-Origin Requests (CORS)

In `agri_ui/app.py`, update your CORS config to allow your Vercel domain:

```python
from flask_cors import CORS
CORS(app, supports_credentials=True, origins=[
    "http://localhost:5173",
    "https://your-app.vercel.app",   # ← replace with your Vercel URL
])
```

Commit and push — Railway will auto-redeploy.

---

## Step 6 — Initialise the Database

After Railway deploys, run this once to create all tables:

```bash
railway run python -c "from app import app, db; \
  app.app_context().__enter__(); db.create_all(); print('Done')"
```

Or SSH into Railway shell and run it from there.

---

## Step 7 — Seed Admin Accounts

From Railway shell or locally (with `DATABASE_URL` set):

```bash
python seed_users.py
```

This creates:
- `admin@pqnk.com` / `admin123`
- `superadmin@pqnk.com` / `super123`

> ⚠️ Change these passwords immediately after first login.

---

## Step 8 — Configure Qdrant (Vector Database)

Your Qdrant collection `pqnk_v2` must be populated before the chatbot works.

1. Log in to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster → copy the URL and API key
3. Add them to Railway environment variables
4. Run your ingestion pipeline locally pointing to the cloud Qdrant:
   ```bash
   python ingest_to_postgres.py
   ```

---

## Step 9 — Final Checks

| Check | How |
|---|---|
| Login works | Visit your Vercel URL → login |
| Admin dashboard loads | Login as admin@pqnk.com |
| Chatbot responds | Ask a PQNK question |
| OTP emails arrive | Register a new account |
| Repository upload works | Admin → Content Management |

---

## Cost Summary

| Service | Free Tier |
|---|---|
| **Vercel** (Frontend) | ✅ Unlimited (Hobby plan) |
| **Railway** (Flask + PostgreSQL) | ✅ $5/month credit |
| **Qdrant Cloud** | ✅ 1 cluster, 1GB free |
| **Gmail SMTP** | ✅ Free with App Password |
| **OpenAI API** | ❌ ~$0.01–0.05 per chatbot query |

> The only cost is OpenAI API usage. For an FYP demo with low traffic this will be negligible (a few dollars/month at most).

---

## Local Development (Quick Start)

```bash
# Backend
cd c:\Users\smwaj\fyp_text
python agri_ui\app.py

# Frontend (new terminal)
cd frontend-react\PQNK_Frontend
npm install
npm run dev
```

Visit `http://localhost:5173`
