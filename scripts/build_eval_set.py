"""Freeze relevance labels for the retrieval eval.

``data/eval/seeds.jsonl`` holds hand-written questions. Each seed names the
document(s) that answer it and an ``answer_regex``: a phrase copied from the
passage that actually contains the answer. This script finds every stored
chunk in those documents whose text matches the regex and writes the
``(doc_id, chunk_no, page)`` triples to ``data/eval/questions.jsonl``.

Why regex labels: chunk numbers change whenever the chunker changes, the
answer text does not. Re-run this after re-chunking (``make eval-labels``).
Seeds that match no chunk are reported and fail the run.

Usage:
    uv run python scripts/build_eval_set.py [--tenant-id default]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from agentic_rag_platform.ingest.paths import repo_root
from agentic_rag_platform.ingest.store import connect
from agentic_rag_platform.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = repo_root()
    parser.add_argument("--seeds", type=Path, default=root / "data/eval/seeds.jsonl")
    parser.add_argument("--out", type=Path, default=root / "data/eval/questions.jsonl")
    parser.add_argument("--tenant-id", default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    tenant = args.tenant_id or settings.default_tenant_id

    seeds = [json.loads(line) for line in args.seeds.read_text("utf-8").splitlines() if line]
    doc_ids = sorted({doc for seed in seeds for doc in seed["doc_ids"]})
    with connect(settings.database_url) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, chunk_no, page, content FROM chunks
            WHERE tenant_id = %s AND doc_id = ANY(%s)
            ORDER BY doc_id, chunk_no
            """,
            (tenant, doc_ids),
        ).fetchall()

    missing: list[str] = []
    out_lines: list[str] = []
    for seed in seeds:
        pattern = re.compile(seed["answer_regex"], re.S)
        relevant = [
            {"doc_id": row["doc_id"], "chunk_no": row["chunk_no"], "page": row["page"]}
            for row in rows
            if row["doc_id"] in seed["doc_ids"] and pattern.search(row["content"])
        ]
        if not relevant:
            missing.append(seed["id"])
        record = {key: seed[key] for key in ("id", "type", "lang", "question", "doc_ids")}
        record["relevant"] = relevant
        out_lines.append(json.dumps(record, ensure_ascii=False))

    if missing:
        print(f"No chunk matches the answer_regex for: {', '.join(missing)}", file=sys.stderr)
        return 1
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    total = sum(len(json.loads(line)["relevant"]) for line in out_lines)
    print(f"Wrote {len(out_lines)} questions with {total} relevant chunks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
