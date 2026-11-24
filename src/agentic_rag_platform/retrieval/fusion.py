"""Rank fusion helpers.

Reciprocal Rank Fusion (Cormack et al., 2009):

    score(d) = sum over result lists L of 1 / (k + rank_L(d))

with 1-based ranks and k = 60 by default. RRF only looks at ranks, so BM25
scores (unbounded) and cosine similarities (-1..1) never need to be put on a
common scale. Qdrant implements the same formula server-side
(``FusionQuery(fusion=Fusion.RRF)``); Postgres does it in SQL
(``agentic_rag_platform.retrieval.pg``). This module is the reference
implementation used in tests and for ad-hoc fusion.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf_scores[K: Hashable](rankings: Sequence[Sequence[K]], *, k: int = 60) -> dict[K, float]:
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_fuse[K: Hashable](
    rankings: Sequence[Sequence[K]], *, k: int = 60, limit: int | None = None
) -> list[K]:
    """Fused order; ties keep the order in which keys were first seen."""
    scores = rrf_scores(rankings, k=k)
    first_seen = {key: index for index, key in enumerate(scores)}
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return ordered[:limit] if limit is not None else ordered
