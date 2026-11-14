from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_judge_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Storage
    db_path: str = "data/stylescout.db"
    chroma_path: str = "data/chroma"
    catalog_csv_path: str = "data/synthetic_catalog.csv"

    # Hybrid search / RRF
    # k=60 is the standard RRF constant (Cormack et al. 2009).
    rrf_k: int = 60
    rrf_kw_weight: float = 1.0
    rrf_vec_weight: float = 1.0
    hybrid_n_candidates: int = 50

    # API
    request_timeout_s: int = 30
    max_agent_steps: int = 6
    max_history_turns: int = 10

    # Logging
    log_level: str = "INFO"


settings = Settings()
