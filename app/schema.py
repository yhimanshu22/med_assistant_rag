from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str

class DocumentSource(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[DocumentSource] = []
    total_time: Optional[str] = None
