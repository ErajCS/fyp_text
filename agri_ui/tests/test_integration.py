"""
Integration Tests — verifies Qdrant and OpenAI network connectivity.
Database model tests removed here to avoid PostgreSQL dependency;
they are covered via the Flask test client in test_api.py instead.
"""
import os
import pytest
from qdrant_client import QdrantClient
from openai import OpenAI


def test_qdrant_connectivity():
    """Verify Qdrant is reachable and credentials are valid."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    assert qdrant_url, "QDRANT_URL is missing from environment"
    assert qdrant_key, "QDRANT_API_KEY is missing from environment"

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10.0)
    collections = client.get_collections()
    assert collections is not None


def test_openai_connectivity():
    """Verify OpenAI API key is valid."""
    openai_key = os.getenv("OPENAI_API_KEY")
    assert openai_key, "OPENAI_API_KEY is missing from environment"

    client = OpenAI(api_key=openai_key)
    response = client.models.list()
    assert len(response.data) > 0
