import pytest
# We need to add the parent directory to sys.path so we can import app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app
from app import db

@pytest.fixture
def app():
    # Setup test configuration
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"  # Use in-memory SQLite for tests
    flask_app.config["WTF_CSRF_ENABLED"] = False
    
    with flask_app.app_context():
        # Create all tables in the test database
        db.create_all()
        yield flask_app
        # Teardown the test database
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()
