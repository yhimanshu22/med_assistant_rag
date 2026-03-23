from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import shutil
from contextlib import asynccontextmanager
from time import time
import uvicorn
import asyncio

from med_assistant.models.schemas import QueryRequest, QueryResponse, DocumentSource
from med_assistant.services.rag_service import RAGService
from med_assistant.services.ingestion_service import ingest_documents_generator

rag_service = RAGService()

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
    if not rag_service.qa_chain:
        raise HTTPException(status_code=503, detail="RAG Service not initialized. Check server logs.")

    start_time = time()
    try:
        response = rag_service.answer_question(request.question)
        
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
        raise HTTPException(status_code=500, detail=str(e))

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
