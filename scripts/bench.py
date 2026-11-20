#!/usr/bin/env python3
"""Compare HNSW against exact (sequential) search: latency and recall@10.

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --ef-search 16,40,80 --k 10
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag_platform.ingest.chunker import count_tokens  # noqa: E402
from agentic_rag_platform.ingest.embedder import get_embedder  # noqa: E402
from agentic_rag_platform.ingest.paths import repo_root  # noqa: E402
from agentic_rag_platform.ingest.store import connect, fetch_chunk_texts, search  # noqa: E402
from agentic_rag_platform.ingest.types import SearchHit  # noqa: E402
from agentic_rag_platform.settings import get_settings  # noqa: E402

MARK_START = "<!-- BENCH:START -->"
MARK_END = "<!-- BENCH:END -->"
WARSAW = ZoneInfo("Europe/Warsaw")

# Written queries used alongside excerpts taken from stored chunks. They are
# in English and Polish so the notes can show a realistic mix.
WRITTEN_QUERIES = [
    "What was the ECB Governing Council decision on interest rates?",
    "Euro area inflation outlook and the 2% medium-term target",
    "Financing conditions and credit developments in the euro area",
    "What is the coupon on four-year indexed Polish treasury bonds (COI)?",
    "Warunki emisji trzyletnich obligacji skarbowych TOS",
    "Oprocentowanie rocznych obligacji oszczędnościowych ROR",
    "Family 12-year Polish savings bonds ROD issue letter",
    "External environment and risks to the euro area growth outlook",
    "How is the interest rate on two-year DOR bonds reset?",
    "Emisja emerytalnych dziesięcioletnich obligacji skarbowych EDO",
]


@dataclass(frozen=True)
class QueryCase:
    text: str
    origin: str  # "written" or "chunk:<doc_id>#<chunk_no>"


@dataclass(frozen=True)
class EfResult:
    ef_search: int | None
    recall_at_k: float
    hnsw_ms: list[float]
    exact_ms: list[float]


def _hit_key(hit: SearchHit) -> tuple[str, int]:
    return (hit.doc_id, hit.chunk_no)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def excerpt_from_chunk(text: str, *, rng: random.Random) -> str:
    """Take a mid-chunk window of ~40–80 tokens as a query."""
    words = text.split()
    if len(words) <= 40:
        return text.strip()
    window = rng.randint(40, min(80, len(words)))
    start = rng.randint(0, max(0, len(words) - window))
    return " ".join(words[start : start + window])


def build_queries(hits: list[SearchHit], n: int, rng: random.Random) -> list[QueryCase]:
    written = [QueryCase(text=text, origin="written") for text in WRITTEN_QUERIES]
    generated: list[QueryCase] = []
    # Prefer longer chunks so the excerpt is not the entire chunk.
    candidates = [hit for hit in hits if count_tokens(hit.text) >= 80]
    if not candidates:
        candidates = list(hits)
    rng.shuffle(candidates)
    for hit in candidates:
        generated.append(
            QueryCase(
                text=excerpt_from_chunk(hit.text, rng=rng),
                origin=f"chunk:{hit.doc_id}#{hit.chunk_no}",
            )
        )
        if len(written) + len(generated) >= n:
            break
    combined = written + generated
    if len(combined) < n:
        raise SystemExit(
            f"Need {n} queries but only built {len(combined)}. Ingest more documents first."
        )
    # Keep all written queries, fill the rest from chunks, then trim to n
    # while preserving order: written first.
    return combined[:n]


def run_bench(
    *,
    database_url: str,
    tenant_id: str,
    model_name: str,
    queries: list[QueryCase],
    query_vectors: list[list[float]],
    k: int,
    ef_values: list[int],
) -> list[EfResult]:
    results: list[EfResult] = []
    with connect(database_url) as conn:
        # Exact search once per query (ground truth). Timed separately.
        exact_hits: list[list[SearchHit]] = []
        exact_ms: list[float] = []
        for vector in query_vectors:
            started = time.perf_counter()
            hits = search(
                conn,
                tenant_id=tenant_id,
                query=vector,
                k=k,
                embedding_model=model_name,
                backend="exact",
            )
            exact_ms.append((time.perf_counter() - started) * 1000)
            exact_hits.append(hits)

        for ef in ef_values:
            hnsw_ms: list[float] = []
            recalls: list[float] = []
            for vector, truth in zip(query_vectors, exact_hits, strict=True):
                started = time.perf_counter()
                approx = search(
                    conn,
                    tenant_id=tenant_id,
                    query=vector,
                    k=k,
                    embedding_model=model_name,
                    ef_search=ef,
                    backend="hnsw",
                )
                hnsw_ms.append((time.perf_counter() - started) * 1000)
                truth_ids = {_hit_key(hit) for hit in truth}
                approx_ids = {_hit_key(hit) for hit in approx}
                if not truth_ids:
                    recalls.append(0.0)
                else:
                    recalls.append(len(truth_ids & approx_ids) / len(truth_ids))
            results.append(
                EfResult(
                    ef_search=ef,
                    recall_at_k=statistics.mean(recalls) if recalls else 0.0,
                    hnsw_ms=hnsw_ms,
                    exact_ms=exact_ms,
                )
            )
    return results


def _fmt_ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.mean(values)
    p50 = _percentile(values, 0.50)
    p95 = _percentile(values, 0.95)
    return f"mean {mean:.2f} ms · p50 {p50:.2f} · p95 {p95:.2f}"


def render_results(
    *,
    when: datetime,
    embedder_name: str,
    dimensions: int,
    tenant_id: str,
    n_chunks: int,
    n_queries: int,
    k: int,
    results: list[EfResult],
    hardware: str,
    notes: list[str],
) -> str:
    lines = [
        f"_Measured {when.strftime('%Y-%m-%d %H:%M')} Europe/Warsaw. "
        f"Embedder `{embedder_name}` (dim={dimensions}), tenant `{tenant_id}`, "
        f"{n_chunks} chunks, {n_queries} queries, recall@{k}. {hardware}_",
        "",
        f"| ef_search | recall@{k} | HNSW latency | Exact (seq scan) latency |",
        "| ---: | ---: | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.ef_search} | {item.recall_at_k:.3f} | {_fmt_ms(item.hnsw_ms)} | "
            f"{_fmt_ms(item.exact_ms)} |"
        )
    lines.append("")
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def upsert_results_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(f"{path} is missing {MARK_START} / {MARK_END} markers")
    start = text.index(MARK_START) + len(MARK_START)
    end = text.index(MARK_END)
    new = text[:start] + "\n\n" + block.rstrip() + "\n\n" + text[end:]
    path.write_text(new, encoding="utf-8")


def hardware_line() -> str:
    cpu = "unknown CPU"
    try:
        info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in info.splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = ""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                mem = f", {kb / 1024 / 1024:.1f} GiB RAM"
                break
    except (OSError, ValueError):
        pass
    return f"Host: {cpu}{mem}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HNSW vs exact-search benchmark")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--ef-search", default="16,40,80", help="Comma-separated ef_search values")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown file to update (default: docs/retrieval-notes.md)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    tenant_id = args.tenant or settings.default_tenant_id
    ef_values = [int(part) for part in args.ef_search.split(",") if part.strip()]
    if not ef_values:
        print("No ef_search values given", file=sys.stderr)
        return 1

    embedder = get_embedder(settings)
    print(f"Embedder: {embedder.name} (dim={embedder.dimensions})")

    with connect(settings.database_url) as conn:
        stored = fetch_chunk_texts(conn, tenant_id=tenant_id, embedding_model=embedder.name)
    if len(stored) < args.k:
        print(
            f"Need at least {args.k} chunks in tenant={tenant_id} for model "
            f"{embedder.name}; found {len(stored)}. Run: make ingest",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    cases = build_queries(stored, args.queries, rng)
    print(f"Embedding {len(cases)} queries…")
    vectors = embedder.embed_queries([case.text for case in cases])
    print("Running HNSW vs exact search…")
    results = run_bench(
        database_url=settings.database_url,
        tenant_id=tenant_id,
        model_name=embedder.name,
        queries=cases,
        query_vectors=vectors,
        k=args.k,
        ef_values=ef_values,
    )

    notes = [
        f"{sum(1 for case in cases if case.origin == 'written')} written queries, "
        f"{sum(1 for case in cases if case.origin.startswith('chunk:'))} "
        "excerpts taken from stored chunks (seed="
        f"{args.seed}).",
        "Recall is |HNSW top-k ∩ exact top-k| / |exact top-k| per query, then averaged. "
        "Exact search uses SET LOCAL enable_indexscan/bitmapscan/indexonlyscan = off.",
        "HNSW index: vector_cosine_ops, m=16, ef_construction=64. Query metric: cosine (<=>).",
    ]
    if embedder.name.startswith("hash-"):
        notes.append(
            "This run used the deterministic hash embedder, so these numbers measure "
            "index behaviour only, not retrieval quality of a semantic model."
        )

    when = datetime.now(tz=WARSAW)
    block = render_results(
        when=when,
        embedder_name=embedder.name,
        dimensions=embedder.dimensions,
        tenant_id=tenant_id,
        n_chunks=len(stored),
        n_queries=len(cases),
        k=args.k,
        results=results,
        hardware=hardware_line(),
        notes=notes,
    )
    out = args.out or (repo_root() / "docs" / "retrieval-notes.md")
    upsert_results_block(out, block)
    print(block)
    print(f"Updated {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
