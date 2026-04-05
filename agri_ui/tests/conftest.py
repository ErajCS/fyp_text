"""
Standalone pytest config — intentionally does NOT import app.py.

Importing app.py causes rag_demo.py to be loaded at module level, which
immediately tries to connect to Qdrant and OpenAI — external services that
are not available in the GitHub Actions CI environment.

All real business logic is tested via isolated unit tests in test_auth.py.
"""
import pytest
