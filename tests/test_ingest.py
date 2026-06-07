import pytest
from unittest.mock import patch, MagicMock
from med_assistant.services.ingestion_service import ingest_documents

@patch("med_assistant.services.ingestion_service.glob.glob")
@patch("med_assistant.services.ingestion_service.fitz.open")
@patch("med_assistant.services.ingestion_service.Chroma")
@patch("med_assistant.services.ingestion_service.HuggingFaceEmbeddings")
def test_ingest_documents_success(mock_embeddings, mock_chroma, mock_fitz_open, mock_glob):
    """Test that ingestion works normally when files are present."""
    mock_glob.return_value = ["data/file1.pdf"]

    mock_page = MagicMock()
    mock_page.get_text.return_value = "This is a test medical document."

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = None
    mock_fitz_open.return_value = mock_doc

    ingest_documents()

    mock_glob.assert_called_once()
    mock_fitz_open.assert_called_once_with("data/file1.pdf")
    mock_doc.__enter__.assert_called_once()
    mock_doc.__exit__.assert_called_once()
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
