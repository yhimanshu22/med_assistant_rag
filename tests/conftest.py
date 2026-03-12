import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_rag_service():
    """Mock the RAG service initialization and answer generation to prevent loading heavy LLMs in tests."""
    from med_assistant.api.main import rag_service
    with patch.object(rag_service, 'initialize'), \
         patch.object(rag_service, 'qa_chain', new=True), \
         patch.object(rag_service, 'answer_question') as mock_answer:
            
            # Setup default mock response
            mock_doc = MagicMock()
            mock_doc.page_content = "This is a mock answer from a document."
            mock_doc.metadata = {"source": "mock_file.pdf", "page": 1}
            
            mock_answer.return_value = {
                "result": "Influenza is a viral infection.",
                "source_documents": [mock_doc]
            }
            yield mock_answer

@pytest.fixture
def client(mock_rag_service):
    """Provide a TestClient for FastAPI with lifespan events triggered."""
    from med_assistant.api.main import app
    with TestClient(app) as client:
        yield client

