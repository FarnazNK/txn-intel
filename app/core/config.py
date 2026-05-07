from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "txn-intel"
    env: str = "dev"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "txn_intel"
    postgres_user: str = "txn"
    postgres_password: str = "txn"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    model_dir: str = "./data/models"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-7"

    feature_cache_ttl_seconds: int = 300
    prediction_log_sample_rate: float = 1.0

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
