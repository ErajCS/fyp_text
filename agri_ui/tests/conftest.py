"""
conftest.py — Force SQLite in-memory BEFORE Flask app is imported.

The problem: importing app.py triggers Flask-SQLAlchemy to connect to
PostgreSQL immediately using the DB_ env vars injected from GitHub Secrets.
The fix: override SQLALCHEMY_DATABASE_URI in os.environ BEFORE the import
so Flask-SQLAlchemy uses in-memory SQLite instead.
"""
import os
import sys
import pytest

# ── CRITICAL: Set the test DB URI BEFORE importing app.py ─────────────────
os.environ["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
# Also zero out the individual DB vars so app.py can't build a postgres URI
os.environ["DB_HOST"] = ""
os.environ["DB_NAME"] = ""
os.environ["DB_USER"] = ""
os.environ["DB_PASS"] = ""
os.environ["SECRET_KEY"] = os.environ.get("SECRET_KEY", "test-ci-secret-key")

# ── Now it's safe to import the Flask app ─────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app as flask_app  # noqa: E402
from app import db  # noqa: E402


@pytest.fixture
def app():
    """Sets up an in-memory SQLite database for testing (no PostgreSQL needed)."""
    flask_app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
    })

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()
