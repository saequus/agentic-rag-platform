# Retrieval notes (Day 2)

This note is the artefact for the Day 2 interview questions: how the
ingestion pipeline is built, how we chose the embedding model, how the
pgvector HNSW index behaves against exact search, and what xAI currently
offers for embeddings. Numbers in the results block are produced by
`make bench` (`scripts/bench.py`).

There is **no Google Drive connector** in this drop (explicitly out of
scope for the Day 2 implementation).

## Pipeline

```
PDF (pypdf)
  → structure-aware recursive chunker (512–800 tokens, ~12% overlap,
     section titles kept as metadata, SHA-256 content hash)
  → embedder (default: intfloat/multilingual-e5-base)
  → Postgres + pgvector
       documents (tenant_id, doc_id, title, source_uri, page_count)
       chunks    (tenant_id, doc_id, chunk_no, page, section,
                  content, content_hash, embedding_model, embedding)
       HNSW index on embedding (vector_cosine_ops, m=16, ef_construction=64)
```

Upserts are keyed by `(tenant_id, doc_id, chunk_no)`. If the content hash
and the embedding model are unchanged, the vector is not recomputed. When
a document shrinks, leftover higher `chunk_no` rows are deleted. Search
always filters on `tenant_id` (and on `embedding_model`, so two models
are never mixed in one result set).

Token counts in the chunker use tiktoken `cl100k_base`. That is a stable
approximation, not the e5 tokenizer; a chunk may be a few tokens shorter
or longer once it reaches the model.

## Chunking choices

Chunk size is a retrieval parameter, not a constant to guess. The
starting point here is structure-aware splitting (headings such as
`1 External environment`, `Rozdział 2`, all-caps issue-letter titles)
followed by recursive splits on paragraphs and sentences into a
512–800 token window with 12% overlap (inside the 10–15% band). Short
final leftovers are merged back into the previous chunk when they share
a section and still fit in 800 tokens.

Too-small chunks lose the sentence that made the heading meaningful;
too-large chunks dilute the embedding and waste context later. A common
follow-up (Day 3 / parent-document retrieval) is to *search* small
chunks and *return* the surrounding section. We would only move the
window after measuring recall@k on a labelled set.

## Embeddings

Default is **`intfloat/multilingual-e5-base`** (768-d) via
`sentence-transformers`, because the corpus is mixed Polish / English
public-finance text and the model is small enough to run on a laptop
CPU. Passages are prefixed with `passage: `, queries with `query: `,
and vectors are L2-normalised so cosine and dot-product rankings agree.

`BAAI/bge-m3` (1024-d) is supported as `EMBEDDING_MODEL=BAAI/bge-m3`.
Its dense embeddings are used *without* the e5 prefixes, which matches
the model card; they are still normalised.

`text-embedding-3-small` (1536-d) is available when
`EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` is set, as a quality
and cost comparison — not as the default, because the platform story is
on-prem-capable. Switching models means re-embedding everything; every
row stores `embedding_model` for that reason. The schema's
`vector(dim)` must match the model (`make db-init` infers the
dimension). Mixing dimensions in one table is not supported.

### xAI embeddings (checked 2026-11-20)

xAI is the default **chat / tool-calling** provider in this repo. It
also exposes an embeddings API, but we do not use it as the Day 2
default.

What the current docs and OpenAPI say:

- The official REST surface (`https://api.x.ai/v1`, OpenAPI at
  `https://api.x.ai/api-docs/openapi.json`, Swagger UI at
  `https://api.x.ai/docs/`) includes `POST /v1/embeddings`,
  `GET /v1/embedding-models` and `GET /v1/embedding-models/{model_id}`.
  The OpenAPI example model id is `v1`. Input may be a string or a list
  of up to 128 strings (or token ids). Optional fields include
  `dimensions` and `encoding_format` (`float` or `base64`).
- The schema text says: for best performance, prepend `query: ` in
  front of query content and `passage: ` in front of passage/text —
  the same convention as e5.
- The [mTLS](https://docs.x.ai/developers/advanced-api-usage/mtls)
  guide lists `/v1/embeddings` alongside chat and responses as a
  supported path.
- The [Collections API](https://docs.x.ai/developers/rest-api-reference/collections/collection)
  (launched August 2025) uses `grok-embedding-small` to index uploaded
  files and search them via `POST /v1/documents/search`. That is a
  managed RAG feature, not a drop-in replacement for storing our own
  vectors in pgvector.
- The public [models](https://docs.x.ai/developers/models) page, as of
  this check, lists chat, image, video and voice models and does **not**
  list a dedicated embeddings model in the pricing table. There is also
  no first-class embeddings guide at
  `https://docs.x.ai/developers/guides/embeddings` (404).

So: an embeddings endpoint exists, the collections product uses
`grok-embedding-small`, but a multilingual open model remains the right
default for an on-prem PL/EN finance corpus that we want to be able to
explain line by line.

## Index: HNSW vs IVFFlat vs exact

The index is HNSW with `vector_cosine_ops`, `m=16`,
`ef_construction=64`. Query-time `ef_search` is configurable
(`HNSW_EF_SEARCH`, default 40, which is also pgvector's default).
`make bench` compares that path to an exact search obtained by
`SET LOCAL enable_indexscan/bitmapscan/indexonlyscan = off` and reports
latency and recall@10.

Why HNSW rather than IVFFlat for this project:

- HNSW is a graph index. It needs no training step, takes inserts
  incrementally, and usually gives a better speed/recall trade-off.
  It uses more memory and is slower to build.
- IVFFlat clusters vectors into `lists` and must be built after a
  representative amount of data is loaded. It is lighter in memory but
  recall drops unless `probes` is raised.

Filtered queries (`WHERE tenant_id = …`) can return fewer than k rows
if the index finds neighbours and the filter is applied afterwards.
pgvector 0.8 added iterative index scans for this; this database is
0.8.x (0.8.0 on the Linux box used to develop Day 2, 0.8.7 on the
macOS Postgres.app). We also keep a b-tree on
`(tenant_id, embedding_model)`. Partitioning by tenant is the next
step if a single HNSW graph gets noisy.

pgvector vs a dedicated store (Qdrant, Milvus): at this scale
(thousands of chunks, not tens of millions) one system with
transactions, joins, backups and a tenant column is the right default.
A dedicated engine becomes interesting for native hybrid (sparse+dense)
search, quantization, and sharding — Day 3 adds a hybrid path.

## Corpus

`data/urls.txt` lists **23** public PDFs that were downloaded and
checked for a `%PDF-` magic byte on 2026-11-20:

- 12 ECB Economic Bulletins (2024–2026), from
  `https://www.ecb.europa.eu/pub/pdf/ecbu/`.
- 11 official Polish retail Treasury bond issue letters (OTS, ROR, DOR,
  TOS, COI, EDO, ROD, plus one 2022 premium letter) hosted at
  `https://www.obligacjeskarbowe.pl/`.

NBP Inflation Reports are **not** in the committed list. From the
build network, `nbp.pl` served an Imperva “Pardon Our Interruption”
challenge instead of the PDF, so those official URLs could not be
verified here. They can be added to `data/urls.txt` and downloaded from
a network that can reach nbp.pl (typically a browser session in PL).

Downloaded files go to `data/raw/` and are gitignored.

## Benchmark results

Recall is measured against exact nearest neighbours, not against a
human-labelled relevant set. That answers “does the index find the
same neighbours as a sequential scan?”, which is the HNSW question.
Retrieval *quality* of the embedding model is a separate eval (Day 7).

Queries are a mix of written PL/EN finance questions and mid-chunk
excerpts taken from the stored corpus (see `scripts/bench.py`).

<!-- BENCH:START -->

_Measured 2026-11-20 23:23 Europe/Warsaw. Embedder `intfloat/multilingual-e5-base` (dim=768), tenant `default`, 9104 chunks, 50 queries, recall@10. Host: Intel(R) Xeon(R) Processor, 15.6 GiB RAM._

| ef_search | recall@10 | HNSW latency | Exact (seq scan) latency |
| ---: | ---: | --- | --- |
| 16 | 1.000 | mean 89.45 ms · p50 88.36 · p95 98.40 | mean 79.30 ms · p50 85.30 · p95 91.89 |
| 40 | 1.000 | mean 88.92 ms · p50 87.37 · p95 100.07 | mean 79.30 ms · p50 85.30 · p95 91.89 |
| 80 | 1.000 | mean 89.18 ms · p50 87.14 · p95 99.75 | mean 79.30 ms · p50 85.30 · p95 91.89 |

Notes:
- 10 written queries, 40 excerpts taken from stored chunks (seed=7).
- Recall is |HNSW top-k ∩ exact top-k| / |exact top-k| per query, then averaged. Exact search uses SET LOCAL enable_indexscan/bitmapscan/indexonlyscan = off.
- HNSW index: vector_cosine_ops, m=16, ef_construction=64. Query metric: cosine (<=>).

<!-- BENCH:END -->

At 9,104 chunks the sequential scan is slightly *faster* than HNSW
(~79 ms vs ~89 ms mean) and recall@10 is already 1.0 at `ef_search=16`.
That is expected at this scale: 9k × 768-d vectors fit in memory and a
seq scan is a tight CPU loop, while HNSW pays for graph bookkeeping.
The index becomes the right default as the corpus grows into the
hundreds of thousands or millions of vectors; the interview line is
“I measured both, and I would not claim an HNSW speed-up until the
numbers show one.”


## Commands

```bash
make db-init          # CREATE EXTENSION vector + tables + HNSW
make ingest           # download the corpus (data/raw) and upsert chunks
make bench            # 50 queries, HNSW vs exact, updates this file
```

`POST /ingest` and `POST /search` on the FastAPI app exercise the same
pipeline. `/search` is tenant-scoped and uses the HNSW index.
