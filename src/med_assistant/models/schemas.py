from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = []

class DocumentSource(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: Optional[List[DocumentSource]] = []
    total_time: str
    confidence: Optional[float] = 1.0
    metrics: Optional[Dict[str, float]] = {}
