from unittest.mock import patch

def test_query_endpoint_success(client, mock_rag_service):
    """Test successful query endpoint response."""
    payload = {"question": "What is Influenza?"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["question"] == "What is Influenza?"
    assert data["answer"] == "Influenza is a viral infection."
    assert "total_time" in data
    
    assert len(data["source_documents"]) == 1
    source = data["source_documents"][0]
    assert source["page_content"] == "This is a mock answer from a document."
    assert source["metadata"]["source"] == "mock_file.pdf"
    
    # Ensure our mock was called with the correct question
    mock_rag_service.assert_called_once_with(
        "What is Influenza?", [], enable_evaluation=False
    )

def test_query_endpoint_missing_question(client):
    """Test standard validation error for missing field."""
    payload = {}  # Missing question
    response = client.post("/query", json=payload)
    
    assert response.status_code == 422  # Unprocessable Entity (FastAPI validation)

def test_query_endpoint_uninitialized_service(client):
    """Test when RAG service is not initialized."""
    with patch('med_assistant.api.main.rag_service.llm', None), \
         patch('med_assistant.api.main.rag_service.vectordb', None):
        payload = {"question": "What happens if not initialized?"}
        response = client.post("/query", json=payload)
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()
