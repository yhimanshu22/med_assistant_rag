from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATABASE_URL = f"sqlite:///{(_PROJECT_ROOT / 'users.db').resolve().as_posix()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATA_DIR: str = "data"
    DB_DIR: str = "chroma_db"
    MODEL_CACHE_DIR: str = "models_cache"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    GPU_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.2"
    CPU_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Retrieval tuning (advanced RAG)
    RETRIEVE_K_DENSE: int = 8
    RETRIEVE_K_BM25: int = 12
    RERANK_TOP_N: int = 6
    # Chroma "distance" threshold. Higher typically means less relevant (depends on embedding metric).
    # Used as a safety valve to refuse/ask for clarification when retrieval is weak.
    RETRIEVAL_MAX_DISTANCE: float = 0.9
    # Cross-encoder reranker model for relevance reranking
    RERANKER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Server master switch: if False, users cannot enable Ragas evaluation in the UI.
    # Ragas adds multiple extra LLM calls per answer (very slow on CPU).
    ENABLE_RAG_EVALUATION: bool = True

    # Auth
    DATABASE_URL: str = _DEFAULT_DATABASE_URL
    SECRET_KEY: str = "change-me-in-production-use-a-long-random-secret"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

settings = Settings()
