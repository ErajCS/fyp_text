def test_signup_missing_fields(client):
    """Test that signup without required fields fails gracefully and returns 400."""
    res = client.post("/api/signup", json={})
    assert res.status_code == 400
    assert not res.get_json().get("success")

def test_signup_weak_password(client):
    """Test that signup rejects a weak password."""
    res = client.post("/api/signup", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "simplepassword",   # missing uppercase, digit, special char
        "confirm_password": "simplepassword"
    })
    
    assert res.status_code == 400
    data = res.get_json()
    assert not data["success"]
    assert "Password must contain" in data["message"]

def test_login_invalid_credentials(client):
    """Test that login with invalid credentials properly returns 401 Unauthorized."""
    res = client.post("/api/login", json={
        "email": "nobody@test.com",
        "password": "SomeSecretPassword!123"
    })
    assert res.status_code == 401
    assert not res.get_json().get("success")
