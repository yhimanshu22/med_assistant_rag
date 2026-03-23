# 🩺 Medical Assistant RAG System

This project is a powerful Retrieval-Augmented Generation (RAG) system wrapped into a highly scalable FastAPI application and an interactive Streamlit UI. It is designed to assist users in querying and retrieving precise information from medical PDF documents using a Llama-3 based natural language processing pipeline.

## ✨ Features
*   **Local Document Ingestion**: Process and index any PDF medical documents locally.
*   **High-Performance AI**: Utilizes the state-of-the-art `meta-llama/Meta-Llama-3-8B-Chat` model (loaded in 4-bit precision to save VRAM) for natural, human-like responses.
*   **Vector Database**: Employs **ChromaDB** with sentence embeddings for highly accurate context retrieval.
*   **Dual Architecture**: 
    *   **FastAPI Backend**: A lightweight, asynchronous RESTful API for integrations.
    *   **React + Vite Frontend**: A modern, interactive, and high-performance user interface for querying data and managing documents.

## 🛠️ Tech Stack
*   **Backend & API**: `FastAPI`, `Uvicorn`
*   **Frontend UI**: `React`, `Vite`, `TypeScript`, `Lucide React`, `Framer Motion`
*   **LLM & RAG**: `LangChain`, `HuggingFace Transformers`, `PyTorch` (`bitsandbytes`, `accelerate`, `xformers`)
*   **Embeddings**: `Sentence-Transformers`
*   **Vector Store**: `ChromaDB`
*   **Document Parsers**: `PyPDF`

## 📋 Prerequisites
*   Python 3.10+
*   NVIDIA GPU with CUDA support (Recommended for 4-bit mode loading, requires ~6GB+ VRAM. System can fall back to smaller CPU-based models where possible).
*   A Hugging Face account with access to the `Meta-Llama-3-8B-Chat` model.

## 🚀 Getting Started

### 1. Installation & Setup
Clone the repository and install the required dependencies using `uv`:
```bash
uv sync   # This creates the virtual environment and installs dependencies from pyproject.toml
```

Authenticate your Hugging Face account (to download the model checkpoint):
```bash
huggingface-cli login
```

### 2. Add and Ingest Data
1. Create a `data/` folder in the project's root directory if it does not already exist.
2. Place your reference PDF documents (e.g., medical studies, regulations, or drug manuals) into the `data/` directory.
3. Run the data ingestion script to process the PDFs and initialize the vector database.
```bash
uv run python -m med_assistant.services.ingestion_service
```
> *This will generate a `chroma_db/` directory containing the document embeddings.*

### 3. Run the Application

You'll need two terminal instances to run the full application (Backend + Frontend).

**Start the FastAPI Backend:**
```bash
uv run uvicorn med_assistant.api.main:app --host 0.0.0.0 --port 8000
```

**Start the React UI (In a new terminal):**
```bash
cd frontend
npm install
npm run dev
```
> *The interactive chat interface will be available at `http://localhost:5173`.*

---

## 🐳 Docker Deployment

The project includes a `docker-compose.yml` for simplified local orchestration of both the backend and frontend.

### 1. Build and Start
Ensure you have Docker and Docker Compose installed:
```bash
docker compose up --build
```

### 2. Access the Application
- **Frontend**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`

### 3. Persistence
The following directories are mapped as volumes to ensure data persists across container restarts:
- `./chroma_db`: The vector database index.
- `./data`: Source PDF documents for ingestion.
- `./models_cache`: Cached Hugging Face models to avoid re-downloading.

---

## ☸️ Kubernetes Deployment

Kubernetes manifests are provided in the `k8s/` directory for deploying to a cluster (e.g., Minikube, GKE, EKS).

### 1. Build Images
Tag and build your images (replace with your registry if needed):
```bash
docker build -t med-assistant-backend:latest .
docker build -t med-assistant-frontend:latest ./frontend
```

### 2. Deploy to Cluster
Apply all manifests in the `k8s/` directory:
```bash
kubectl apply -f k8s/
```

### 3. Verify Deployment
Check the status of your pods and services:
```bash
kubectl get pods -n med-assistant
kubectl get svc -n med-assistant
```

### 4. Access the Frontend (Minikube example)
```bash
minikube service frontend-service -n med-assistant
```

---

## 🌐 API Reference

If you want to query the backend programmatically, you can use the `/query` endpoint.

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "question": "What are the common symptoms of Influenza?"
}
```

**Response Example:**
```json
{
  "question": "What are the common symptoms of Influenza?",
  "answer": "Common symptoms of influenza include fever, chills, muscle aches, cough, congestion, runny nose, headaches, and fatigue.",
  "source_documents": [
    {
      "page_content": "...symptoms may include sudden onset of fever, chills, and muscle aches...",
      "metadata": {"source": "data/influenza_guide.pdf", "page": 4}
    }
  ],
  "total_time": "2.45 sec"
}
```

**Testing:**
The repository includes a `test_api.py` script to test connectivity to the FastAPI backend manually. You can run it via:
```bash
python test_api.py
```
