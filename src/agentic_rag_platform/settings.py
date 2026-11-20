"""Application settings loaded from the environment and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["xai", "openai", "anthropic"]
EmbeddingProvider = Literal["local", "openai", "hash"]

DEFAULT_XAI_REST_BASE = "https://api.x.ai/v1"
DEFAULT_XAI_GRPC_HOST = "api.x.ai"


class Settings(BaseSettings):
    """Runtime configuration.

    `LLM_PROVIDER` selects the adapter. xAI is the default because that is the
    key this project is developed against. OpenAI and Anthropic adapters are
    implemented so their tool-calling and structured-output APIs can be learned
    and compared (see `docs/learning-notes.md`).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: ProviderName = "xai"

    xai_api_key: SecretStr | None = None
    xai_base_url: str = DEFAULT_XAI_REST_BASE
    xai_model: str = "grok-4.7"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"

    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tool_iterations: int = Field(default=5, ge=1, le=20)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    tool_http_timeout_seconds: float = Field(default=8.0, gt=0)

    database_url: str = "postgresql://slava@localhost:5432/agentic_rag"

    # Day 2 embeddings. "local" is a SentenceTransformer (default: multilingual-e5-base).
    # "openai" uses text-embedding-3-small and requires OPENAI_API_KEY.
    # "hash" is a deterministic unit-vector backend for tests / index micro-benchmarks.
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "intfloat/multilingual-e5-base"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    default_tenant_id: str = "default"
    hnsw_ef_search: int = Field(default=40, ge=1, le=800)

    @property
    def model_name(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.xai_model

    @property
    def xai_api_host(self) -> str:
        """gRPC host for the official `xai-sdk` (`api_host`).

        `XAI_BASE_URL` is documented as the OpenAI-compatible REST base
        (`https://api.x.ai/v1`). The first-party SDK talks gRPC to `api.x.ai`.
        """
        parsed = urlparse(self.xai_base_url)
        return parsed.hostname or DEFAULT_XAI_GRPC_HOST


@lru_cache
def get_settings() -> Settings:
    return Settings()
