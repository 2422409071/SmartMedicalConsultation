"""
Global Configuration Module
Reads configuration from .env file using pydantic_settings
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    All settings can be overridden via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== LLM Configuration =====
    model_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="LLM API base URL (DashScope compatible with OpenAI)"
    )
    model_api_key: str = Field(
        default="",
        description="LLM API key (DashScope API Key)"
    )
    model_name: str = Field(
        default="qwen-plus",
        description="LLM model name"
    )

    # ===== Neo4j Configuration =====
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j database connection URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="12345678",
        description="Neo4j password"
    )

    # ===== Embedding Model Configuration =====
    embedding_model_path: str = Field(
        default="D:/models/bge-m3",
        description="Local path to BGE-M3 embedding model"
    )

    # ===== Logging Configuration =====
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # ===== Application Configuration =====
    app_name: str = Field(
        default="Medical Consultation Assistant",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI server port"
    )
    frontend_port: int = Field(
        default=8501,
        description="Streamlit frontend port"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures we only load settings once.
    """
    return Settings()


# Global settings instance
settings = get_settings()
