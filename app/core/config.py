from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    groq_api_key: str = ""

    summary_model: str = "qwen/qwen3.6-27b"

    answer_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store / docstore ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "multimodal_rag"

    cohere_api_key: str = ""

    redis_url: str = "redis://localhost:6379"
    use_redis: bool = True
    local_docstore_path: str = "./data/docstore/store.json"
    #----conversational memory ---------
    sqlite_db_path: str = "./data/chat_memory.db"
    memory_turns: int = 6

    # --- Ingestion ---
    upload_dir: str = "./data/uploads"
    extract_image: str = "./data/extracted_images"
    pdf_parse_strategy: str = "hi_res"  # unstructured strategy
    chunk_max_chars: int = 4000
    chunk_combine_under_chars: int = 2000
    max_concurrent_summaries: int = 2

    # --- Retrieval ---
    retrieval_k: int = 5


@lru_cache
def get_settings()->Settings:
    return Settings()




    