"""Application settings loaded from the environment and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["xai", "openai", "anthropic"]
EmbeddingProvider = Literal["local", "openai", "hash"]
RetrievalStrategy = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
RetrievalBackend = Literal["pgvector", "qdrant"]
FusionMethod = Literal["rrf", "dbsf"]
LexicalScoring = Literal["idf", "ts_rank_cd"]
CheckpointerKind = Literal["postgres", "memory"]

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

    # Day 3 retrieval. Strategy and backend are defaults; /search can override
    # both per request. "sparse" is BM25-style lexical search (Postgres
    # full-text on pgvector, a BM25 sparse vector on Qdrant).
    retrieval_strategy: RetrievalStrategy = "hybrid"
    retrieval_backend: RetrievalBackend = "pgvector"
    # How many candidates each leg (dense, sparse) contributes before fusion,
    # and how many fused candidates the cross-encoder rescores.
    retrieval_candidates: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    qdrant_fusion: FusionMethod = "rrf"
    # Postgres lexical scorer: "idf" (BM25-flavoured, default) or the naive
    # "ts_rank_cd" baseline (no IDF). See retrieval/pg.py.
    pg_lexical_scoring: LexicalScoring = "idf"
    # Qdrant: QDRANT_URL wins (server, e.g. http://localhost:6333). Otherwise
    # QDRANT_PATH is used in embedded local mode (":memory:" for tests).
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "chunks"
    # Cross-encoder reranker. MiniLM is the CPU-friendly default;
    # BAAI/bge-reranker-v2-m3 is multilingual (Polish) but ~5x larger.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_batch_size: int = Field(default=16, ge=1, le=256)

    # Day 4 LangGraph agent (POST /chat, /chat/stream, /chat/resume).
    # "postgres" persists thread state in DATABASE_URL (tables are created by
    # AsyncPostgresSaver.setup()); "memory" is per-process (tests, quick demos).
    agent_checkpointer: CheckpointerKind = "postgres"
    # Loop / cost guards. recursion_limit caps graph super-steps; max_rewrites
    # caps the retrieve -> grade -> rewrite loop; max_llm_calls is the per-turn
    # budget kept in graph state; node timeout bounds every LLM call.
    agent_recursion_limit: int = Field(default=25, ge=5, le=200)
    agent_max_rewrites: int = Field(default=2, ge=0, le=5)
    agent_max_llm_calls: int = Field(default=10, ge=2, le=50)
    agent_node_timeout_seconds: float = Field(default=45.0, gt=0)
    agent_top_k: int = Field(default=5, ge=1, le=20)
    # Thread memory: how many previous turns (user + assistant pairs) go into prompts.
    agent_history_turns: int = Field(default=4, ge=0, le=20)
    # Retrieval defaults for the agent (fall back to RETRIEVAL_STRATEGY / RETRIEVAL_BACKEND).
    agent_retrieval_strategy: RetrievalStrategy | None = None
    agent_retrieval_backend: RetrievalBackend | None = None

    # OpenTelemetry. Official OTEL_* env vars are honoured. Tracing stays off
    # until an exporter is configured (OTLP endpoint and/or console).
    otel_sdk_disabled: bool = False
    otel_service_name: str = "agentic-rag-platform"
    otel_environment: str = "local"
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_protocol: str = "http/protobuf"
    otel_traces_exporter: str = "otlp"
    otel_traces_sampler: str = "parentbased_traceidratio"
    otel_traces_sampler_arg: float = Field(default=1.0, ge=0.0, le=1.0)
    otel_redact_content: bool = True

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
