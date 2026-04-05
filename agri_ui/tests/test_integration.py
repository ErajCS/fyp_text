"""
Comprehensive Integration Tests for PQNK
These tests verify that the external dependencies (OpenAI, Qdrant, DB)
are correctly configured and reachable in the environment.
"""
import os
import pytest
from qdrant_client import QdrantClient
from openai import OpenAI
import time

def test_qdrant_connectivity():
    """Verify Qdrant is reachable and credentials are valid."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    assert qdrant_url, "QDRANT_URL is missing from environment"
    assert qdrant_key, "QDRANT_API_KEY is missing from environment"
    
    # Attempt to connect to Qdrant cluster
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10.0)
    collections = client.get_collections()
    # It just needs to not throw an Unauthorized exception
    assert collections is not None

def test_openai_connectivity():
    """Verify OpenAI API key is valid."""
    openai_key = os.getenv("OPENAI_API_KEY")
    assert openai_key, "OPENAI_API_KEY is missing from environment"
    
    client = OpenAI(api_key=openai_key)
    # Perform a light network call to ensure key isn't revoked or invalid
    response = client.models.list()
    assert len(response.data) > 0

def test_database_model(app):
    """Test that SQLAlchemy models and the database session work."""
    from app import db, User, Resource
    
    with app.app_context():
        # Create a test user
        new_user = User(
            name="Integration Test",
            email="integration@pqnk.com",
            phone="1234567890",
            password="hashedpassword123",
            role="admin",
            designation="Tester",
            organization="PQNK"
        )
        db.session.add(new_user)
        db.session.commit()
        
        # Verify user was saved
        saved_user = User.query.filter_by(email="integration@pqnk.com").first()
        assert saved_user is not None
        assert saved_user.name == "Integration Test"
        
        # Cleanup
        db.session.delete(saved_user)
        db.session.commit()
