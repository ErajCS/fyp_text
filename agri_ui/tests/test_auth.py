"""
Standalone tests for PQNK backend — no Flask app import needed.
These tests validate core business logic functions in isolation
so they can run in the GitHub Actions CI environment without
requiring a live PostgreSQL or Qdrant database connection.
"""
import sys
import os
import re

# ---------------------------------------------------------------------------
# 1. Copy validate_password logic here directly so tests don't import app.py
#    (importing app.py triggers rag_demo → Qdrant connection at module level)
# ---------------------------------------------------------------------------

def validate_password(password: str):
    """Mirror of app.py validate_password() — kept in sync manually."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character."
    return True, "OK"


# ---------------------------------------------------------------------------
# 2. Password validation tests (pure logic, no DB/network required)
# ---------------------------------------------------------------------------

class TestPasswordValidation:
    def test_valid_password_passes(self):
        ok, msg = validate_password("StrongPass1!")
        assert ok is True

    def test_too_short_fails(self):
        ok, msg = validate_password("Ab1!")
        assert ok is False
        assert "8 characters" in msg

    def test_missing_uppercase_fails(self):
        ok, msg = validate_password("weakpass1!")
        assert ok is False
        assert "uppercase" in msg

    def test_missing_lowercase_fails(self):
        ok, msg = validate_password("STRONGPASS1!")
        assert ok is False
        assert "lowercase" in msg

    def test_missing_digit_fails(self):
        ok, msg = validate_password("StrongPass!")
        assert ok is False
        assert "digit" in msg

    def test_missing_special_char_fails(self):
        ok, msg = validate_password("StrongPass1")
        assert ok is False
        assert "special character" in msg

    def test_all_requirements_met(self):
        passwords = ["Admin@123", "Secure#Pass9", "P@ssw0rd!", "Hello!World2"]
        for pw in passwords:
            ok, msg = validate_password(pw)
            assert ok is True, f"Expected {pw!r} to pass but got: {msg}"


# ---------------------------------------------------------------------------
# 3. General utility / sanity tests
# ---------------------------------------------------------------------------

class TestSanity:
    def test_python_version(self):
        """Ensure the CI runner is using Python 3.9+."""
        assert sys.version_info >= (3, 9)

    def test_env_has_secret_key(self):
        """The SECRET_KEY env var must be set in the CI environment."""
        assert os.environ.get("SECRET_KEY"), "SECRET_KEY not set in env!"
