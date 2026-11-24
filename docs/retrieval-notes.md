# Retrieval notes (Days 2–3)

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

_Measured 2026-11-21 12:00 Europe/Warsaw. Embedder `intfloat/multilingual-e5-base` (dim=768), tenant `default`, 9104 chunks, 50 queries, recall@10. Host: unknown CPU._

| ef_search | recall@10 | HNSW latency | Exact (seq scan) latency |
| ---: | ---: | --- | --- |
| 16 | 0.742 | mean 2.33 ms · p50 2.29 · p95 3.03 | mean 28.42 ms · p50 27.88 · p95 29.30 |
| 40 | 0.916 | mean 2.90 ms · p50 2.83 · p95 3.31 | mean 28.42 ms · p50 27.88 · p95 29.30 |
| 80 | 1.000 | mean 28.19 ms · p50 27.68 · p95 28.89 | mean 28.42 ms · p50 27.88 · p95 29.30 |

Notes:
- 10 written queries, 40 excerpts taken from stored chunks (seed=7).
- Recall is |HNSW top-k ∩ exact top-k| / |exact top-k| per query, then averaged. Exact search uses SET LOCAL enable_indexscan/bitmapscan/indexonlyscan = off.
- HNSW index: vector_cosine_ops, m=16, ef_construction=64. Query metric: cosine (<=>).

<!-- BENCH:END -->

At 9,104 chunks the sequential scan costs ~28 ms and HNSW ~3 ms, but
HNSW at `ef_search=40` only finds ~92% of the true top 10 on these
queries (and ~82% on the Day 3 eval questions, see below). At
`ef_search=80` "recall 1.000" is not the index getting better: the
latency jumps to the seq-scan figure because **the planner stops using
the HNSW index** (its cost estimate grows with `ef_search`) and runs an
exact scan instead. Check with `EXPLAIN ANALYZE`, not just with recall.
At this scale exact search is a perfectly good choice; HNSW earns its
keep in the hundreds of thousands to millions of vectors. Interview
line: "I measured both, and I would not claim an HNSW speed-up until
the numbers show one."

## Day 3: hybrid search, Qdrant, reranking

### What is implemented

| strategy | pgvector backend | Qdrant backend |
|---|---|---|
| `dense` | HNSW cosine on `chunks.embedding` | named vector `dense` (HNSW m=16, ef_construct=64) |
| `sparse` | `tsvector` (`'simple'` config) + IDF-weighted `ts_rank` | sparse vector `bm25`, `Modifier.IDF` |
| `hybrid` | dense top 50 + lexical top 50, **RRF in one SQL query** | Query API: 2 × `Prefetch` + `RrfQuery` (or DBSF) |
| `hybrid_rerank` | hybrid top 50 → cross-encoder → top k | same |

`RETRIEVAL_STRATEGY` / `RETRIEVAL_BACKEND` set the defaults; `POST /search`
accepts `strategy` and `backend` per request and returns `timings_ms`
(`embed_ms`, `search_ms`, `rerank_ms`). Qdrant is loaded from Postgres
(`make qdrant-sync`) **reusing the stored e5 vectors**, so both engines index
exactly the same chunks and vectors.

Code: `src/agentic_rag_platform/retrieval/` (`sparse.py` tokenizer + BM25
vectors, `fusion.py` reference RRF, `pg.py`, `qdrant_index.py`, `rerank.py`,
`service.py`), `migrations/002_fulltext.sql`.

### Reciprocal Rank Fusion

```
RRF(d) = Σ_lists  1 / (k + rank_list(d))        rank is 1-based, k = 60
```

- Uses **ranks, not scores**. BM25 scores are unbounded, cosine is in [-1, 1];
  RRF never has to normalise them onto one scale.
- `k` damps the head: with k=60, rank 1 is worth 1/61 ≈ 0.0164 and rank 10 is
  1/70 ≈ 0.0143, so agreement between lists matters more than one list's top spot.
  A document in both lists at rank 5 (2/65 ≈ 0.031) beats one that is #1 in only one.
- Missing from a list = contributes 0 (the SQL uses a `FULL OUTER JOIN` + `COALESCE`).
- Alternatives: weighted RRF (Qdrant `Rrf(k, weights)`), score fusion after
  min-max or **DBSF** (distribution-based: normalise each list by mean ± 3σ,
  then sum; `QDRANT_FUSION=dbsf`), or a learned fusion / reranker.
- Each leg is `ORDER BY … LIMIT 50` *before* `row_number()`, so Postgres can
  still use HNSW and GIN; a window function over the whole table would not.

**Fusion only helps when both legs are decent.** With the naive Postgres
lexical scorer (`ts_rank_cd`, no IDF) the sparse leg scored hit@5 0.24 and
hybrid (0.42) was barely better than dense alone (0.40) on pgvector. With
IDF weighting the sparse leg went to 0.58 and hybrid to 0.64.

### Postgres full-text, and why IDF matters

- `to_tsvector('simple', content)`: lowercase + split, no stemming, no stopwords.
  The corpus is English (ECB) **and Polish** (treasury bonds); Postgres ships no
  Polish dictionary/stemmer, so `'english'` would only help half the corpus and
  mangle the other half. Stopwords (EN + PL) are removed from the query in Python.
- `ts_rank` / `ts_rank_cd` have **no inverse document frequency**: "rate"
  (in thousands of chunks) counts as much as "EDO1136" (in one). Measured:
  `ts_rank_cd` OR-query hit@5 **0.24**.
- `PG_LEXICAL_SCORING=idf` (default): per query term count `df` through the GIN
  index, `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`, score = Σ idf(t) · ts_rank(t).
  hit@5 **0.58**, and faster (~30 ms p50 vs ~75 ms) because it never ranks one
  huge OR match set.
- Gotcha worth telling in an interview: the first version took **2.4 s per
  query**. `EXPLAIN ANALYZE` showed Postgres had inlined the CTE and re-ran the
  per-term `count(*)` for each of 2,683 candidate rows. `WITH … AS MATERIALIZED`
  fixed it (~14 ms when measured alone).
- Real BM25 in Postgres: ParadeDB `pg_search` or `pg_textsearch`. Without an
  extension, the IDF trick above gets most of the way.

### Qdrant layout and BM25 sparse vectors

- One collection, two named vectors: `dense` (768-d cosine) and sparse `bm25`.
- Document side stores the BM25 term-frequency part
  `tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))` (k1 = 1.2, b = 0.75, avgdl
  computed at sync time, 51.5 tokens here); `Modifier.IDF` makes Qdrant apply IDF
  at query time from live collection stats, so adding documents never requires
  re-weighting old points. Query side is 1.0 per unique term.
- Term ids are a stable 31-bit blake2b hash of the token (same idea as fastembed's
  `Qdrant/bm25`). We use our own tokenizer so pgvector and Qdrant see the same
  tokens; fastembed `Qdrant/bm25` (English stemming) or **bge-m3 sparse**
  (learned, multilingual) are drop-in alternatives worth evaluating.
- Hybrid query: `query_points(prefetch=[Prefetch(dense, limit=50, filter),
  Prefetch(bm25, limit=50, filter)], query=RrfQuery(Rrf(k=60)), filter)`.
- Embedded mode (`QDRANT_PATH`, no server) is brute force and ignores payload
  indexes; fine for tests and a laptop, not for latency numbers. The eval used a
  real Qdrant 1.20.2 server.

### Multi-tenant filtering in Qdrant

- Payload index on `tenant_id` with **`is_tenant=true`** (keyword index). Qdrant
  then stores each tenant's points together, so a tenant's query reads its data
  sequentially instead of seeking across the whole segment. HNSW is
  payload-aware for indexed fields, so filtered queries keep their recall. With
  many tenants, `hnsw_config: {m: 0, payload_m: 16}` drops the global graph and
  builds per-tenant graphs only (faster indexing; cross-tenant queries get slow,
  which is fine because there should be none).
- **IDF leaks across tenants by default.** With `Modifier.IDF` (and BM25 text
  search) document frequencies are shard-wide, so one tenant's corpus changes
  another tenant's term weights, and in principle reveals what is common in it.
  Every sparse query here passes `params.idf.corpus = <tenant filter>` so the
  statistics come from the caller's tenant only
  (`test_qdrant_idf_statistics_are_scoped_to_the_tenant`: global IDF scores the
  two test chunks 3.09 vs 0.05, tenant-scoped IDF scores them equally).
- The tenant filter is **always added server-side** (`tenant_filter()` in
  `qdrant_index.py`, applied to every prefetch *and* the outer query) from the
  caller's tenant, which in production comes from the auth context (JWT claim,
  API key → tenant), never from a request body or a user-supplied filter.
- Options as tenants grow: one collection + `is_tenant` index (many small
  tenants; default), user-defined sharding (**shard keys**) to give big tenants
  their own shard, **tiered multitenancy** (small tenants share a fallback shard,
  a tenant is promoted to a dedicated shard at roughly 20k points), or a
  collection per very large / regulated tenant (strong isolation, separate
  snapshots and deletes, more overhead). Avoid one collection per small tenant;
  Qdrant Cloud caps a cluster at 1,000 collections by default.
  Source: [Qdrant multitenancy guide](https://qdrant.tech/documentation/guides/multiple-partitions/).
- pgvector equivalent: `WHERE tenant_id = $1` on every query (here) and, for
  defence in depth, Postgres **row-level security** keyed on a session variable.

### Bi-encoder vs cross-encoder: measured latency

CPU only (this box, 8 vCPU, no GPU), same 50 questions:

| step | model | p50 | p95 |
|---|---|---|---|
| embed 1 query (bi-encoder) | `intfloat/multilingual-e5-base` | 15.0 ms | 24.6 ms |
| vector search (engine only) | Qdrant dense / pgvector HNSW | 8.2 ms / 11.9 ms | |
| rerank 50 pairs (cross-encoder) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | 560–744 ms (≈315 ms in a separate single-model run) | 914 ms |
| rerank 50 pairs (cross-encoder) | `BAAI/bge-reranker-v2-m3` (568M, multilingual) | 10,333 ms (≈8,000 ms in a separate single-model run) | 14,032 ms |

- A bi-encoder embeds passages **once at ingest**; query time is one forward pass
  plus an ANN lookup. A cross-encoder runs **one forward pass per (query,
  passage) pair**, all layers attending across both texts: much better ranking,
  O(candidates) cost on every request. That is why it only rescores the top 50.
- On CPU the reranker is ~95% of request latency. Levers: fewer candidates
  (20–30), a GPU (tens of ms for 50 pairs), ONNX/int8, shorter `max_length`,
  or a hosted rerank API. Measure recall@5 vs candidates before cutting.
- MiniLM is trained on English MS MARCO; it still helped Polish questions
  with exact identifiers (pgvector exact/pl-bonds hit@5 0.36 → 0.55). The
  multilingual `bge-reranker-v2-m3` is better (Qdrant hybrid_rerank hit@5
  0.68 → 0.72, MRR@10 0.56 → 0.65, exact/pl-bonds
  0.46 → 0.64) but ~25× slower on CPU: a GPU decision, not a default.

### Eval: method

- `data/eval/seeds.jsonl`: 50 hand-written questions (32 ECB, 18 Polish
  bonds; 29 "exact" with an identifier/date/number, 21 paraphrases; 8 English
  questions about Polish documents to test cross-lingual retrieval). Each seed
  names the answering document(s) and an `answer_regex` copied from the passage
  that holds the answer.
- `scripts/build_eval_set.py` (`make eval-labels`) marks every chunk in those
  documents that matches the regex as relevant (72 relevant chunks; a fact
  repeated in a summary and the body gives 2). Labels survive re-chunking: re-run
  the script. Questions were written by reading the stored chunks, so they share
  more vocabulary with the text than real user questions would; that favours
  the lexical leg. Treat the numbers as a regression baseline, not a benchmark.
- Metrics: **hit@5** (≥1 relevant chunk in top 5), **recall@5** (share of
  relevant chunks in top 5), **MRR@10**, and **doc_hit@5** (any chunk of the
  right document in top 5).
- Query vectors are computed once and reused, so `search_ms` is engine time only.

### Eval: results (CPU, 2026-11-24)

Full table: [docs/eval-results.md](eval-results.md) (`make eval`).

| backend | strategy | hit@5 | recall@5 | MRR@10 | doc_hit@5 | search p50 ms |
|---|---|---|---|---|---|---|
| pgvector | dense | 0.40 | 0.33 | 0.32 | 0.78 | 11.9 |
| pgvector | sparse | 0.58 | 0.51 | 0.46 | 0.80 | 27.5 |
| pgvector | hybrid | 0.64 | 0.58 | 0.50 | 0.78 | 23.0 |
| pgvector | hybrid_rerank | 0.70 | 0.64 | 0.57 | 0.84 | 31.4 |
| qdrant | dense | 0.54 | 0.44 | 0.41 | 0.88 | 8.2 |
| qdrant | sparse | 0.56 | 0.50 | 0.47 | 0.82 | 7.8 |
| qdrant | hybrid | 0.64 | 0.55 | 0.47 | 0.82 | 14.8 |
| qdrant | hybrid_rerank | 0.68 | 0.62 | 0.56 | 0.82 | 16.6 |

Extra runs (same questions):

| run | hit@5 | recall@5 | MRR@10 | doc_hit@5 | latency p50 |
|---|---|---|---|---|---|
| pgvector sparse, naive `ts_rank_cd` lexical (`PG_LEXICAL_SCORING=ts_rank_cd`) | 0.24 | 0.20 | 0.17 | 0.64 | search 75.4 ms |
| pgvector hybrid, naive `ts_rank_cd` lexical (`PG_LEXICAL_SCORING=ts_rank_cd`) | 0.42 | 0.35 | 0.28 | 0.66 | search 43.9 ms |
| pgvector dense, exact scan (`HNSW_EF_SEARCH=100` → planner picks seq scan) | 0.54 | 0.44 | 0.41 | 0.88 | search 51.4 ms |
| Qdrant hybrid_rerank with `BAAI/bge-reranker-v2-m3` ([results](eval-results-bge-reranker.md)) | 0.72 | 0.67 | 0.65 | 0.88 | rerank 10333 ms |

Latencies were measured on a shared 8-vCPU box with other jobs running (load average 5–9), so treat them as relative, not absolute. The separate single-model reranker timings are in the latency table above.

### What the numbers say

1. **Hybrid + rerank is the best configuration on both engines**: hit@5
   0.40 → 0.70 on pgvector, 0.54 → 0.68 on Qdrant, MRR@10 0.32 → 0.57 and
   0.41 → 0.56.
2. **Identical vectors, different dense results.** pgvector dense 0.40 vs Qdrant
   dense 0.54: pgvector HNSW at `ef_search=40` found only ~82% of the exact top 10
   for these questions. Forcing exact search (`HNSW_EF_SEARCH=100`, where the
   planner switches to a seq scan) gives 0.54, the same as Qdrant, at ~50 ms.
   Lesson: report ANN recall against exact search on *real* queries.
3. **Exact identifiers need the sparse leg**: on "exact" ECB questions (dates,
   numbers) dense pgvector got 0.44 and IDF-sparse 0.72.
4. **Polish bond letters are the hard part** (exact questions hit@5 ≤ 0.64 even with the multilingual reranker, paraphrases ~0):
   the 11 letters share almost all boilerplate, EDO1036 and EDO1136 letters are
   near-identical, and the chunk that says "stopa procentowa wynosi 5,35%" does
   not contain the series code "EDO1136" (that is in chunk 1). doc_hit@5 is much
   higher than hit@5 for the same reason. The fix is **contextual retrieval**:
   prepend a short document header (series code, issue letter number, sale
   month) to each chunk before embedding and BM25 indexing (Anthropic's
   "Contextual Retrieval"), or parent-document retrieval. That is the next
   measured experiment.
5. Chunks are smaller than the 512–800 token target from Day 2 (mean 429
   characters, median 210; the ECB contents pages split into many tiny
   fragments). Worth fixing in the chunker, then re-running `make eval-labels`
   and `make eval`.

## Commands

```bash
make db-init          # CREATE EXTENSION vector + tables + HNSW
make ingest           # download the corpus (data/raw) and upsert chunks
make bench            # 50 queries, HNSW vs exact, updates this file
make qdrant-sync      # Day 3: Postgres chunks + vectors -> Qdrant (dense + BM25)
make eval-labels      # Day 3: seeds.jsonl -> questions.jsonl (relevance labels)
make eval             # Day 3: recall@5 / MRR@10 per strategy -> docs/eval-results.md
```

`POST /ingest` and `POST /search` on the FastAPI app exercise the same
pipeline. `/search` is tenant-scoped; `strategy` and `backend` pick the
retrieval path per request.
