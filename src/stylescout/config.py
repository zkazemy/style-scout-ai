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

    # API
    request_timeout_s: int = 30
    max_agent_steps: int = 6
    max_history_turns: int = 10

    # Logging
    log_level: str = "INFO"


settings = Settings()
