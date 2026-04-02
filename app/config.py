from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Override any stale shell env vars with values from .env
load_dotenv(override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM / Embedding APIs
    anthropic_api_key: str
    openai_api_key: str
    pokemon_tcg_api_key: str = ""

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    es_index_name: str = "pokemon_cards"
    embedding_dim: int = 1536

    # Agent behaviour
    max_retrieve_iterations: int = 3
    min_relevant_docs: int = 3
    top_k_retrieve: int = 10


settings = Settings()
