"""
Flask API Route Tests
Tests the REST endpoints for HTTP status codes and correct JSON formatting.
"""
import json

def test_signup_validation(client):
    """Test that the signup endpoint catches weak passwords."""
    response = client.post(
        '/api/signup',
        json={
            'name': 'API Test User',
            'email': 'api@test.com',
            'phone': '1234567890',
            'password': 'weak',  # Should trigger validation failure
            'role': 'user',
            'designation': 'tester',
            'organization': 'PQNK'
        }
    )
    # The API returns 400 when validation fails
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Password must be at least' in data['message']


def test_missing_fields_signup(client):
    """Test that missing required fields return an error."""
    response = client.post(
        '/api/signup',
        json={'email': 'missing@test.com'}
    )
    assert response.status_code == 400
    assert 'Name, email, and password are required' in json.loads(response.data)['message']


def test_login_missing_user(client):
    """Test logging in with a user that doesn't exist."""
    response = client.post(
        '/api/login',
        json={
            'email': 'doesnotexist@pqnk.com',
            'password': 'ValidPassword123!'
        }
    )
    assert response.status_code == 401
    assert 'Invalid email or password' in json.loads(response.data)['message']


def test_protected_route_unauthorized(client):
    """Ensure the user API endpoint blocks unauthenticated requests."""
    response = client.get('/api/user')
    # Because there's no active session
    assert response.status_code == 401
