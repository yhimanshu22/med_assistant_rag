import pytest
from unittest.mock import patch, MagicMock
from med_assistant.services.ingestion_service import ingest_documents

@patch("med_assistant.services.ingestion_service.glob.glob")
@patch("med_assistant.services.ingestion_service.pypdf.PdfReader")
@patch("med_assistant.services.ingestion_service.Chroma")
@patch("med_assistant.services.ingestion_service.HuggingFaceEmbeddings")
def test_ingest_documents_success(mock_embeddings, mock_chroma, mock_pdf_reader, mock_glob):
    """Test that ingestion works normally when files are present."""
    # Setup mocks
    mock_glob.return_value = ["data/file1.pdf"]
    
    # Mock PDF reader with 1 page containing text
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a test medical document."
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader_instance
    
    # Run
    ingest_documents()
    
    # Assert
    mock_glob.assert_called_once()
    mock_pdf_reader.assert_called_once_with("data/file1.pdf")
    mock_embeddings.assert_called_once()
    mock_chroma.from_documents.assert_called_once()
    mock_chroma.from_documents.return_value.persist.assert_called_once()

@patch("med_assistant.services.ingestion_service.glob.glob")
def test_ingest_documents_no_files(mock_glob, capsys):
    """Test ingestion behavior when no PDF files are found."""
    mock_glob.return_value = []
    
    ingest_documents()
    
    captured = capsys.readouterr()
    assert "No PDF files found" in captured.out
