"""Embedding backends.

Default is a local multilingual SentenceTransformer (e5-base, with bge-m3 as
an option). OpenAI ``text-embedding-3-small`` is used only when
``OPENAI_API_KEY`` is set and ``EMBEDDING_PROVIDER=openai``. A deterministic
hash embedder is included for tests and for index-only benchmarks.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentic_rag_platform.settings import Settings

DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-base"
BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

KNOWN_DIMENSIONS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-m3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "hash-embedder-v1": 768,
}


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


def dimension_for_model(model_name: str, default: int = 768) -> int:
    if model_name in KNOWN_DIMENSIONS:
        return KNOWN_DIMENSIONS[model_name]
    # Allow "hash-embedder-v1:384" style overrides in tests.
    if ":" in model_name:
        maybe_dim = model_name.rsplit(":", 1)[1]
        if maybe_dim.isdigit():
            return int(maybe_dim)
    return default


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider == "openai":
        key = settings.openai_api_key
        if key is None or not key.get_secret_value():
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return OpenAIEmbedder(
            api_key=key.get_secret_value(),
            model=settings.openai_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if provider == "hash":
        return HashEmbedder(dimensions=dimension_for_model(settings.embedding_model))
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


class SentenceTransformerEmbedder:
    """Local multilingual embedder via sentence-transformers.

    ``intfloat/multilingual-e5-*`` models require the ``query: `` / ``passage: ``
    prefixes and are L2-normalised so cosine and dot-product rankings agree.
    ``BAAI/bge-m3`` dense embeddings are used without those prefixes (per the
    model card) and are also normalised.
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, batch_size: int = 16):
        # Imported lazily so unit tests and ``GET /health`` do not load torch.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        dimension_of = getattr(self._model, "get_embedding_dimension", None)
        if dimension_of is None:
            dimension_of = self._model.get_sentence_embedding_dimension
        self._dimensions = int(dimension_of() or 0)
        if self._dimensions <= 0:
            raise RuntimeError(f"Could not read embedding dimension for {model_name}")
        self._use_e5_prefixes = _is_e5_model(model_name)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [_passage_text(text, self._use_e5_prefixes) for text in texts]
        return self._encode(prefixed)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [_query_text(text, self._use_e5_prefixes) for text in texts]
        return self._encode(prefixed)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.astype(float).tolist() for vector in vectors]


class OpenAIEmbedder:
    """OpenAI ``text-embedding-3-small`` (or another embeddings model)."""

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL, batch_size: int = 64):
        from openai import OpenAI

        self._model = model
        self._batch_size = batch_size
        self._client = OpenAI(api_key=api_key)
        self._dimensions = dimension_for_model(model, default=1536)

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(model=self._model, input=batch)
            # The API may return rows out of order; sort by index.
            rows = sorted(response.data, key=lambda row: row.index)
            for row in rows:
                out.append(_l2_normalize([float(x) for x in row.embedding]))
        return out


class HashEmbedder:
    """Deterministic unit vectors for tests and index-only micro-benchmarks.

    Similar strings do **not** land near each other; do not use this to judge
    retrieval quality, only to exercise the store and the HNSW index.
    """

    def __init__(self, dimensions: int = 768, name: str = "hash-embedder-v1"):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        values: list[float] = []
        block = 0
        while len(values) < self._dimensions:
            digest = hashlib.sha256(f"{block}:{text}".encode()).digest()
            # 32 bytes → 16 signed 16-bit values in (-1, 1).
            for i in range(0, 32, 2):
                raw = int.from_bytes(digest[i : i + 2], "big", signed=True)
                values.append(raw / 32768.0)
                if len(values) >= self._dimensions:
                    break
            block += 1
        return _l2_normalize(values[: self._dimensions])


def _is_e5_model(model_name: str) -> bool:
    return bool(re.search(r"(?:^|/)(?:multilingual-)?e5-", model_name, re.IGNORECASE))


def _passage_text(text: str, use_e5: bool) -> str:
    stripped = text.strip()
    if use_e5 and not stripped.lower().startswith("passage:"):
        return f"passage: {stripped}"
    return stripped


def _query_text(text: str, use_e5: bool) -> str:
    stripped = text.strip()
    if use_e5 and not stripped.lower().startswith("query:"):
        return f"query: {stripped}"
    return stripped


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]
