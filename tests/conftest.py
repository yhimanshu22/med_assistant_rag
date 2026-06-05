import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_rag_service():
    """Mock the RAG service initialization and answer generation to prevent loading heavy LLMs in tests."""
    from med_assistant.api.main import rag_service
    with patch.object(rag_service, 'initialize'), \
         patch.object(rag_service, 'llm', new=MagicMock()), \
         patch.object(rag_service, 'vectordb', new=MagicMock()), \
         patch.object(rag_service, 'answer_question') as mock_answer:

        mock_answer.return_value = {
            "answer": "Influenza is a viral infection.",
            "sources": [
                {
                    "page_content": "This is a mock answer from a document.",
                    "metadata": {"source": "mock_file.pdf", "page": 1},
                }
            ],
            "confidence": 1.0,
            "metrics": {"faithfulness": 1.0, "relevance": 1.0},
        }
        yield mock_answer

@pytest.fixture
def client(mock_rag_service):
    """Provide a TestClient for FastAPI with lifespan events triggered."""
    from med_assistant.api.main import app
    from med_assistant.api.deps import get_current_user
    from med_assistant.models.user import User

    def mock_current_user():
        return User(id=1, email="test@example.com", hashed_password="hashed")

    app.dependency_overrides[get_current_user] = mock_current_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
