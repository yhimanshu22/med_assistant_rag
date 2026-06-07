from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
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

from med_assistant.api.auth import router as auth_router
from med_assistant.api.deps import get_current_user
from med_assistant.api.middleware import RequestContextMiddleware
from med_assistant.db.database import init_db
from med_assistant.models.schemas import QueryRequest, QueryResponse, DocumentSource
from med_assistant.models.user import User
from med_assistant.core.config import settings
from med_assistant.core.observability import configure_logging, init_sentry, metrics_registry
from med_assistant.services.rag_service import RAGService
from med_assistant.services.ingestion_service import ingest_documents_generator

rag_service = RAGService()

# Async ingestion job store (process-local)
_ingest_jobs: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    init_sentry(settings.SENTRY_DSN, environment=settings.APP_ENV)
    init_db()
    # Load model and chain on startup
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"Failed to initialize RAG service: {e}")
    yield
    # Clean up if needed

app = FastAPI(title="Medical Assistant RAG API", lifespan=lifespan)
app.include_router(auth_router)
app.add_middleware(RequestContextMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_endpoint():
    vectordb_ready = bool(rag_service.vectordb)
    llm_ready = bool(rag_service.llm)
    ready = vectordb_ready and (llm_ready or settings.LAZY_LLM_LOAD)
    return {
        "status": "ready" if ready else "starting",
        "llm_ready": llm_ready,
        "vectordb_ready": vectordb_ready,
        "lazy_llm_load": settings.LAZY_LLM_LOAD,
        "evaluation_available": settings.ENABLE_RAG_EVALUATION,
    }


@app.get("/metrics")
async def metrics_endpoint():
    """In-process metrics: latency, retrieval hit rate, evaluation scores, errors."""
    return metrics_registry.snapshot()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, _: User = Depends(get_current_user)):
    """
    Endpoint to query the RAG medical assistant.
    """
    if not rag_service.vectordb or (not rag_service.llm and not settings.LAZY_LLM_LOAD):
        raise HTTPException(status_code=503, detail="RAG Service not initialized. Check server logs.")

    start_time = time()
    try:
        response = rag_service.answer_question(
            request.question,
            request.chat_history,
            enable_evaluation=request.enable_evaluation,
        )
        
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
            confidence=response.get('confidence', 0.0),
            metrics=response.get('metrics', {}),
            evaluation_enabled=response.get('evaluation_enabled', False),
        )
    except Exception as e:
        metrics_registry.record_error(event="api.query.failed", error=str(e), path="/query")
        logger.error(f"Error in query endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest, _: User = Depends(get_current_user)):
    """
    Streams answer chunks as newline-delimited JSON (NDJSON).
    """
    if not rag_service.vectordb or (not rag_service.llm and not settings.LAZY_LLM_LOAD):
        raise HTTPException(status_code=503, detail="RAG Service not initialized. Check server logs.")

    async def generate():
        loop = asyncio.get_running_loop()

        def run_stream() -> List[str]:
            lines: List[str] = []
            for item in rag_service.answer_question_stream(
                request.question,
                request.chat_history,
                enable_evaluation=request.enable_evaluation,
            ):
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
async def ingest_endpoint(_: User = Depends(get_current_user)):
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
async def ingest_async_endpoint(_: User = Depends(get_current_user)):
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
async def ingest_status_endpoint(job_id: str, _: User = Depends(get_current_user)):
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...), _: User = Depends(get_current_user)):
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
