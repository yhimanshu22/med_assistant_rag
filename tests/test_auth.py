import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from med_assistant.api.deps import get_current_user
from med_assistant.api.main import app
from med_assistant.db.database import Base, engine
from med_assistant.models.user import User


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def auth_client():
    with patch("med_assistant.api.main.rag_service.initialize"), \
         patch("med_assistant.api.main.rag_service.llm", new=MagicMock()), \
         patch("med_assistant.api.main.rag_service.vectordb", new=MagicMock()):
        with TestClient(app) as client:
            yield client


def _signup_and_login(client: TestClient, email: str = "user@example.com", password: str = "secret123"):
    signup = client.post("/auth/signup", json={"email": email, "password": password})
    assert signup.status_code == 201
    token = signup.json()["access_token"]
    return token


def test_signup_success(auth_client):
    response = auth_client.post("/auth/signup", json={"email": "new@example.com", "password": "password1"})
    assert response.status_code == 201
    data = response.json()
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "new@example.com"
    assert data["access_token"]


def test_signup_duplicate_email(auth_client):
    auth_client.post("/auth/signup", json={"email": "dup@example.com", "password": "password1"})
    response = auth_client.post("/auth/signup", json={"email": "dup@example.com", "password": "password2"})
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()


def test_login_success(auth_client):
    auth_client.post("/auth/signup", json={"email": "login@example.com", "password": "password1"})
    response = auth_client.post("/auth/login", json={"email": "login@example.com", "password": "password1"})
    assert response.status_code == 200
    assert response.json()["user"]["email"] == "login@example.com"


def test_login_invalid_credentials(auth_client):
    auth_client.post("/auth/signup", json={"email": "badlogin@example.com", "password": "password1"})
    response = auth_client.post("/auth/login", json={"email": "badlogin@example.com", "password": "wrong"})
    assert response.status_code == 401


def test_me_requires_auth(auth_client):
    response = auth_client.get("/auth/me")
    assert response.status_code == 401


def test_me_returns_current_user(auth_client):
    token = _signup_and_login(auth_client, "me@example.com")
    response = auth_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


def test_logout_success(auth_client):
    token = _signup_and_login(auth_client, "logout@example.com")
    response = auth_client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_query_requires_auth(auth_client):
    response = auth_client.post("/query", json={"question": "test"})
    assert response.status_code == 401
