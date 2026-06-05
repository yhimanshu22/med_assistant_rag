from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional


class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    email: EmailStr


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


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
