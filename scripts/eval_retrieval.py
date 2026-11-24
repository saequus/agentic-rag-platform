"""Offline retrieval eval: recall@5, hit@5 and MRR@10 per strategy and backend.

    uv run python scripts/eval_retrieval.py                       # pgvector + qdrant
    uv run python scripts/eval_retrieval.py --backends pgvector   # no Qdrant needed

Each question in ``data/eval/questions.jsonl`` lists its relevant chunks
(``scripts/build_eval_set.py``). For every (backend, strategy) pair we run all
questions and report:

* hit@5     share of questions with at least one relevant chunk in the top 5
* recall@5  share of each question's relevant chunks found in the top 5, averaged
* MRR@10    mean of 1 / rank of the first relevant chunk (0 if not in the top 10)
* doc_hit@5 share of questions with any chunk of the right document in the top 5
            (looser: the answer document was found, maybe not the answer chunk)

Query vectors are computed once and reused, so ``search_ms`` is the engine
only; ``embed_ms`` (bi-encoder, one query) and ``rerank_ms`` (cross-encoder,
N candidates) are reported separately. Latencies are wall-clock on whatever
machine runs this; one warm-up query per configuration is not timed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_rag_platform.ingest.embedder import get_embedder
from agentic_rag_platform.ingest.paths import repo_root
from agentic_rag_platform.retrieval.service import Retriever
from agentic_rag_platform.settings import RetrievalBackend, RetrievalStrategy, get_settings

STRATEGIES: tuple[RetrievalStrategy, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass
class Score:
    hit5: list[float] = field(default_factory=list)
    recall5: list[float] = field(default_factory=list)
    rr10: list[float] = field(default_factory=list)
    doc_hit5: list[float] = field(default_factory=list)
    search_ms: list[float] = field(default_factory=list)
    rerank_ms: list[float] = field(default_factory=list)
    by_group: dict[str, list[float]] = field(default_factory=dict)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def group_of(question: dict[str, Any]) -> str:
    corpus = "pl-bonds" if question["doc_ids"][0].startswith("pl-bond") else "ecb"
    return f"{question['type']}/{corpus}"


def evaluate(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    vectors: list[list[float]],
    backend: RetrievalBackend,
    strategy: RetrievalStrategy,
    tenant_id: str,
) -> Score:
    score = Score()
    # warm-up (connection pool, model graph, OS caches)
    retriever.retrieve(
        questions[0]["question"],
        tenant_id=tenant_id,
        k=10,
        strategy=strategy,
        backend=backend,
        query_vector=vectors[0],
    )
    for question, vector in zip(questions, vectors, strict=True):
        result = retriever.retrieve(
            question["question"],
            tenant_id=tenant_id,
            k=10,
            strategy=strategy,
            backend=backend,
            query_vector=vector,
        )
        relevant = {(item["doc_id"], item["chunk_no"]) for item in question["relevant"]}
        ranked = [(hit.doc_id, hit.chunk_no) for hit in result.hits]
        top5 = ranked[:5]
        found = len(relevant.intersection(top5))
        score.hit5.append(1.0 if found else 0.0)
        right_docs = set(question["doc_ids"])
        score.doc_hit5.append(1.0 if any(doc in right_docs for doc, _ in top5) else 0.0)
        score.recall5.append(found / len(relevant))
        rr = 0.0
        for rank, key in enumerate(ranked[:10], start=1):
            if key in relevant:
                rr = 1.0 / rank
                break
        score.rr10.append(rr)
        score.by_group.setdefault(group_of(question), []).append(1.0 if found else 0.0)
        score.search_ms.append(result.timings_ms.get("search_ms", 0.0))
        if "rerank_ms" in result.timings_ms:
            score.rerank_ms.append(result.timings_ms["rerank_ms"])
    return score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval")
    root = repo_root()
    parser.add_argument("--questions", type=Path, default=root / "data/eval/questions.jsonl")
    parser.add_argument("--backends", default="pgvector,qdrant")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--json-out", type=Path, default=root / "data/eval/results.json")
    parser.add_argument("--md-out", type=Path, default=root / "docs/eval-results.md")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.rerank_model:
        settings = settings.model_copy(update={"rerank_model": args.rerank_model})
    tenant = args.tenant_id or settings.default_tenant_id
    questions = [
        json.loads(line) for line in args.questions.read_text("utf-8").splitlines() if line
    ]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    started = time.perf_counter()
    embedder = get_embedder(settings)
    load_s = time.perf_counter() - started
    retriever = Retriever(settings=settings, embedder=embedder)
    embedder.embed_queries([questions[0]["question"]])  # warm-up
    vectors: list[list[float]] = []
    embed_ms: list[float] = []
    for question in questions:
        started = time.perf_counter()
        vectors.append(embedder.embed_queries([question["question"]])[0])
        embed_ms.append((time.perf_counter() - started) * 1000)
    print(f"Embedder {embedder.name} loaded in {load_s:.1f}s; {len(questions)} queries embedded")

    rows: list[dict[str, Any]] = []
    for backend in backends:
        for strategy in strategies:
            try:
                score = evaluate(retriever, questions, vectors, backend, strategy, tenant)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 - report and continue with other configs
                print(f"{backend}/{strategy}: FAILED: {exc}", file=sys.stderr)
                continue
            row = {
                "backend": backend,
                "strategy": strategy,
                "hit@5": round(_mean(score.hit5), 3),
                "recall@5": round(_mean(score.recall5), 3),
                "mrr@10": round(_mean(score.rr10), 3),
                "doc_hit@5": round(_mean(score.doc_hit5), 3),
                "search_ms_p50": round(_pct(score.search_ms, 0.5), 1),
                "search_ms_p95": round(_pct(score.search_ms, 0.95), 1),
                "rerank_ms_p50": round(_pct(score.rerank_ms, 0.5), 1) if score.rerank_ms else None,
                "rerank_ms_p95": round(_pct(score.rerank_ms, 0.95), 1) if score.rerank_ms else None,
                "hit@5_by_group": {
                    g: round(_mean(v), 3) for g, v in sorted(score.by_group.items())
                },
            }
            rows.append(row)
            print(json.dumps(row))

    groups = sorted({group_of(q) for q in questions})
    group_counts = {g: sum(1 for q in questions if group_of(q) == g) for g in groups}
    meta = {
        "questions": len(questions),
        "relevant_chunks": sum(len(q["relevant"]) for q in questions),
        "groups": group_counts,
        "embedder": embedder.name,
        "reranker": settings.rerank_model,
        "candidates": settings.retrieval_candidates,
        "rrf_k": settings.rrf_k,
        "qdrant_fusion": settings.qdrant_fusion,
        "qdrant": settings.qdrant_url or settings.qdrant_path,
        "hnsw_ef_search": settings.hnsw_ef_search,
        "embed_ms_p50": round(_pct(embed_ms, 0.5), 1),
        "embed_ms_p95": round(_pct(embed_ms, 0.95), 1),
        "machine": f"{platform.system()} {platform.machine()}, CPU only",
        "date": time.strftime("%Y-%m-%d"),
    }
    args.json_out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n", "utf-8")
    args.md_out.write_text(render_markdown(meta, rows, groups), "utf-8")
    print(f"Wrote {args.json_out} and {args.md_out}")
    return 0


def render_markdown(meta: dict[str, Any], rows: list[dict[str, Any]], groups: list[str]) -> str:
    lines = [
        "# Retrieval eval results",
        "",
        "Generated by `make eval` (`scripts/eval_retrieval.py`). Do not edit by hand.",
        "",
        f"- {meta['questions']} questions, {meta['relevant_chunks']} relevant chunks; "
        + ", ".join(f"{g}: {n}" for g, n in meta["groups"].items()),
        f"- embedder `{meta['embedder']}`, reranker `{meta['reranker']}` "
        f"(top {meta['candidates']} -> top 10), RRF k={meta['rrf_k']}, "
        f"Qdrant fusion {meta['qdrant_fusion']}, pgvector hnsw.ef_search={meta['hnsw_ef_search']}",
        f"- {meta['machine']}, {meta['date']}; query embedding (bi-encoder, 1 query) "
        f"p50 {meta['embed_ms_p50']} ms / p95 {meta['embed_ms_p95']} ms",
        "",
        "| backend | strategy | hit@5 | recall@5 | MRR@10 | doc_hit@5 | search p50 / p95 ms "
        "| rerank p50 / p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rerank = f"{row['rerank_ms_p50']} / {row['rerank_ms_p95']}" if row["rerank_ms_p50"] else "–"
        lines.append(
            f"| {row['backend']} | {row['strategy']} | {row['hit@5']:.2f} | "
            f"{row['recall@5']:.2f} | {row['mrr@10']:.2f} | {row['doc_hit@5']:.2f} | "
            f"{row['search_ms_p50']} / {row['search_ms_p95']} | {rerank} |"
        )
    lines += [
        "",
        "hit@5 by question group:",
        "",
        "| backend | strategy | " + " | ".join(groups) + " |",
        "|---|---|" + "---|" * len(groups),
    ]
    for row in rows:
        cells = " | ".join(f"{row['hit@5_by_group'].get(g, 0):.2f}" for g in groups)
        lines.append(f"| {row['backend']} | {row['strategy']} | {cells} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
