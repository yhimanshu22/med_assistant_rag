# Medical Assistant RAG API - Deployment Guide

This project wraps a Llama-3 based RAG system into a FastAPI application.

## Prerequisites
*   Python 3.10+
*   NVIDIA GPU with CUDA support (Optional, but recommended. Falls back to TinyLlama on CPU)
*   HuggingFace Account (for accessing Llama-3 model)

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Data**
    *   Create a `data` folder if it doesn't exist.
    *   Place your PDF documents (e.g., `uiact_final_draft.pdf` or medical docs) into the `data/` folder.

3.  **Ingest Data**
    Run the ingestion script to process PDFs and create the vector database:
    ```bash
    python ingest.py
    ```
    This will create a `chroma_db` directory.

4.  **Run the Server**
    Start the FastAPI application:
    ```bash
    python main.py
    ```
    Or using uvicorn directly:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Usage

**Endpoint**: `POST /query`

**Request Body**:
```json
{
  "question": "What is the EU AI Act?"
}
```

**Response**:
```json
{
  "question": "What is the EU AI Act?",
  "answer": "The EU AI Act is ...",
  "source_documents": [
    {
      "page_content": "...",
      "metadata": {"source": "data/aiact.pdf", "page": 10}
    }
  ],
  "total_time": "5.123 sec"
}
```

## Notes
*   **Model Source**: The code uses `meta-llama/Meta-Llama-3-8B-Chat`. Ensure you have access to this model on Hugging Face and are logged in (`huggingface-cli login`).
*   **GPU Memory**: The model is loaded in 4-bit mode to save memory, but still requires a decent GPU (approx 6GB+ VRAM).
