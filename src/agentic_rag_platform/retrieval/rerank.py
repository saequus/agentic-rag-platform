"""Cross-encoder reranking.

A bi-encoder (e5) embeds query and passage separately, so passages are
embedded once at ingest and search is a vector lookup. A cross-encoder reads
``[query, passage]`` together through every transformer layer: far more
accurate, but one forward pass per candidate at query time, so it only ever
rescores the top-N candidates from the first stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Protocol, runtime_checkable

from agentic_rag_platform.ingest.types import SearchHit

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, batch_size: int = 16) -> None:
        # Lazy import: tests and /health never load torch.
        from sentence_transformers import CrossEncoder

        self._name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=512)

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        return [float(value) for value in scores]


def rerank(
    reranker: Reranker, query: str, hits: Sequence[SearchHit], top_k: int
) -> list[SearchHit]:
    """Rescore ``hits`` with the cross-encoder and keep the best ``top_k``."""
    if not hits:
        return []
    scores = reranker.score(query, [hit.text for hit in hits])
    if len(scores) != len(hits):
        raise RuntimeError("reranker returned the wrong number of scores")
    order = sorted(range(len(hits)), key=lambda index: -scores[index])
    return [replace(hits[index], score=scores[index]) for index in order[:top_k]]


@lru_cache(maxsize=2)
def get_reranker(model_name: str, batch_size: int = 16) -> Reranker:
    return CrossEncoderReranker(model_name, batch_size=batch_size)
