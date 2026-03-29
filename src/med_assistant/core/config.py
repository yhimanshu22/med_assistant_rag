from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATA_DIR: str = "data"
    DB_DIR: str = "chroma_db"
    MODEL_CACHE_DIR: str = "models_cache"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    GPU_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Chat"
    CPU_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Groq Settings
    GROQ_API_KEY: str | None = None
    GROQ_MODEL_ID: str = "llama-3.3-70b-versatile"
    USE_GROQ: bool = True
    
    class Config:

        env_file = ".env"

settings = Settings()
