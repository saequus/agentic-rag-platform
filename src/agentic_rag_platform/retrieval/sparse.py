"""Lexical side of hybrid search.

One tokenizer feeds both backends so dense-vs-sparse comparisons are fair:

* Postgres: the query terms become an OR ``websearch_to_tsquery('simple', ...)``
  against a ``to_tsvector('simple', content)`` column (no stemming, because the
  corpus mixes English and Polish and Postgres ships no Polish dictionary).
* Qdrant: chunks become BM25 sparse vectors. The document side stores the
  saturated term frequency ``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``;
  Qdrant multiplies by IDF at query time (``Modifier.IDF``), so the collection
  never needs re-indexing when the corpus grows. The query side is 1.0 per term.

Term ids are a stable 31-bit hash of the token, the same trick fastembed's
``Qdrant/bm25`` model uses; collisions are rare enough at this vocabulary size.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

BM25_K1 = 1.2
BM25_B = 0.75

# Keep decimal numbers and identifiers together: "5,35", "0.5", "edo1136".
_TOKEN = re.compile(r"\w+(?:[.,]\w+)*", re.UNICODE)

# Small English + Polish stopword list. Postgres 'simple' keeps every word, and
# ts_rank has no IDF, so very common words would otherwise dominate an OR query.
_STOPWORDS_EN = """
    a an and are as at be by did do does for from had has have how in into is it its of
    on or that the their this to was were what when where which who why will with can
    could about after before between than then there these those under over per our we
    you your
"""
_STOPWORDS_PL = """
    i w z ze na od o się jest są nie że jak jakie jaka jaki jakim jakich ile czy oraz
    dla po przez które który która których także lub albo ten ta te tego tej tym jego
    jej ich być było może mogą
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_EN.split() + _STOPWORDS_PL.split())


def tokenize(text: str) -> list[str]:
    """Lowercase, NFKC-normalise and split into word/number tokens."""
    normalised = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _TOKEN.finditer(normalised):
        token = match.group(0).strip("_")
        if not token or token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def query_terms(text: str) -> list[str]:
    """Unique query terms in first-seen order."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        seen.setdefault(token, None)
    return list(seen)


def term_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def average_length(texts: Iterable[str]) -> float:
    lengths = [len(tokenize(text)) for text in texts]
    if not lengths:
        return 1.0
    return max(1.0, sum(lengths) / len(lengths))


def bm25_document_vector(
    text: str,
    *,
    avgdl: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> SparseVector:
    """Document-side BM25 weights (IDF applied by the engine)."""
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])
    counts = Counter(term_id(token) for token in tokens)
    length_norm = 1.0 - b + b * (len(tokens) / max(avgdl, 1.0))
    indices: list[int] = []
    values: list[float] = []
    for index, tf in sorted(counts.items()):
        indices.append(index)
        values.append(tf * (k1 + 1.0) / (tf + k1 * length_norm))
    return SparseVector(indices=indices, values=values)


def bm25_query_vector(text: str) -> SparseVector:
    ids = sorted({term_id(token) for token in tokenize(text)})
    return SparseVector(indices=ids, values=[1.0] * len(ids))


def websearch_or_query(text: str) -> str | None:
    """Build an OR query for ``websearch_to_tsquery``; ``None`` if no terms."""
    terms = query_terms(text)
    if not terms:
        return None
    return " or ".join(terms)
