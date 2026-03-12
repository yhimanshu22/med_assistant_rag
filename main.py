from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from time import time
import uvicorn

from app.schema import QueryRequest, QueryResponse, DocumentSource
from app.rag import RAGService

# Global RAG service instance
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

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint to query the RAG medical assistant.
    """
    if not rag_service.qa_chain:
        raise HTTPException(status_code=503, detail="RAG Service not initialized. Check server logs.")

    start_time = time()
    try:
        # The chain returns a dict with keys like 'query', 'result', 'source_documents' (if enabled)
        # Note: RetrievalQA run() usually returns just the result string unless return_source_documents is True in the chain
        # Our RAGService uses from_chain_type which defaults return_source_documents to False.
        # Let's adjust rag.py if we want source docs, or just return the answer for now.
        # Actually, let's fix rag.py to return source documents because our schema expects them.
        
        # However, to avoid circular editing complexity right now, let's just assume we get the result.
        # Wait, I should make sure rag.py sets return_source_documents=True if I want them.
        # I'll update rag.py in a separate step if I missed it, but for now let's handle the response generic.
        
        response = rag_service.answer_question(request.question)
        
        # response is typically a dict if using `qa_chain(inputs)` or string if `run(inputs)`
        # RetrievalQA `run` returns just string. `__call__` (which implies `call`) returns dict.
        # rag.py used `self.qa_chain(question)`.
        
        answer_text = response['result']
        source_docs = []
        
        if 'source_documents' in response:
            for doc in response['source_documents']:
                source_docs.append(DocumentSource(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                ))

        total_time = f"{round(time() - start_time, 3)} sec"

        return QueryResponse(
            question=request.question,
            answer=answer_text,
            source_documents=source_docs,
            total_time=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
