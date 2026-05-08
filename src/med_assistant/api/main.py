from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import shutil
from contextlib import asynccontextmanager
from time import time
import uvicorn
import asyncio
import traceback
import logging
import json
import uuid
import threading
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

from med_assistant.models.schemas import QueryRequest, QueryResponse, DocumentSource
from med_assistant.services.rag_service import RAGService
from med_assistant.services.ingestion_service import ingest_documents_generator

rag_service = RAGService()

# Async ingestion job store (process-local)
_ingest_jobs: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and chain on startup
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"Failed to initialize RAG service: {e}")
    yield
    # Clean up if needed

app = FastAPI(title="Medical Assistant RAG API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint to query the RAG medical assistant.
    """
    if not rag_service.llm or not rag_service.vectordb:
        raise HTTPException(status_code=503, detail="RAG Service not initialized. Check server logs.")

    start_time = time()
    try:
        response = rag_service.answer_question(request.question, request.chat_history)
        
        answer_text = response.get('answer', "No answer generated.")
        source_docs = []
        
        for doc in response.get('sources', []):
            source_docs.append(DocumentSource(
                page_content=doc['page_content'],
                metadata=doc['metadata']
            ))

        total_time = f"{round(time() - start_time, 3)} sec"

        return QueryResponse(
            question=request.question,
            answer=answer_text,
            source_documents=source_docs,
            total_time=total_time,
            confidence=response.get('confidence', 1.0),
            metrics=response.get('metrics', {})
        )
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streams answer chunks as newline-delimited JSON (NDJSON).
    """
    if not rag_service.llm or not rag_service.vectordb:
        raise HTTPException(status_code=503, detail="RAG Service not initialized. Check server logs.")

    async def generate():
        loop = asyncio.get_running_loop()

        def run_stream() -> List[str]:
            lines: List[str] = []
            for item in rag_service.answer_question_stream(request.question, request.chat_history):
                lines.append(json.dumps(item))
            return lines

        try:
            lines = await loop.run_in_executor(None, run_stream)
            for line in lines:
                yield line + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/ingest")
async def ingest_endpoint():
    """
    Endpoint to trigger document ingestion and stream progress.
    """
    async def generate():
        loop = asyncio.get_running_loop()
        # Since ingestion_service is synchronous and does heavy lifting, we yield its results back to avoid blocking totally
        for msg in ingest_documents_generator():
            yield msg + "\n"
        
        # After ingestion logic has yielded all its parts, refresh the RAG model pipeline
        yield "\nReloading Vector Database indices...\n"
        try:
            # We run the initialization on the threadpool safely
            await loop.run_in_executor(None, rag_service.initialize)
            yield "Database Refresh Complete.\n"
        except Exception as e:
            yield f"Error refreshing Database: {e}\n"

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/ingest/async")
async def ingest_async_endpoint():
    """
    Starts ingestion in the background and returns a job id.
    Client can poll `/ingest/{job_id}` for status + logs.
    """
    job_id = uuid.uuid4().hex
    _ingest_jobs[job_id] = {"status": "running", "logs": [], "started_at": time(), "ended_at": None, "error": None}

    def run_job():
        try:
            for msg in ingest_documents_generator():
                _ingest_jobs[job_id]["logs"].append(msg)
            _ingest_jobs[job_id]["logs"].append(json.dumps({"step": "refresh", "message": "Reloading Vector Database indices..."}))
            rag_service.initialize()
            _ingest_jobs[job_id]["logs"].append(json.dumps({"step": "refresh", "message": "Database Refresh Complete."}))
            _ingest_jobs[job_id]["status"] = "completed"
        except Exception as e:
            _ingest_jobs[job_id]["status"] = "failed"
            _ingest_jobs[job_id]["error"] = str(e)
            _ingest_jobs[job_id]["logs"].append(json.dumps({"step": "error", "message": str(e), "status": "error"}))
        finally:
            _ingest_jobs[job_id]["ended_at"] = time()

    t = threading.Thread(target=run_job, daemon=True)
    t.start()

    return {"job_id": job_id, "status": "running"}

@app.get("/ingest/{job_id}")
async def ingest_status_endpoint(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload a medical PDF document.
    """
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("med_assistant.api.main:app", host="0.0.0.0", port=8000, reload=False)
