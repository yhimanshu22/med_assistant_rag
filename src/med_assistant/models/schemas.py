from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str

class DocumentSource(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: Optional[List[DocumentSource]] = []
    total_time: str
