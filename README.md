# agentic-rag-platform

On-prem-capable **agentic RAG** reference platform: a FastAPI gateway in front
of a LangGraph agent, MCP tools, hybrid retrieval (pgvector + Qdrant +
reranker), local inference (vLLM / NVIDIA NIM), Langfuse tracing, evals in CI,
and a Helm chart.

This repo is the portfolio project for a Senior AI Platform Engineer
(Generative & Agentic AI) role. The rule is simple: **claim only what you can
demo**. The 20-day roadmap below is the build plan; Day 1 is what runs today.

Default LLM provider is **xAI Grok** via the official [`xai-sdk`](https://pypi.org/project/xai-sdk/)
(`grok-4.7`). OpenAI and Anthropic adapters are included so their tool-calling
and structured-output APIs can be learned and compared — see
[docs/learning-notes.md](docs/learning-notes.md).

## What works today (Days 1–2 + OpenTelemetry lab)

- FastAPI app with `GET /health`, `POST /chat`, `POST /ingest`, `POST /search`
- Provider interface (`Protocol`) with **xAI as default**, plus OpenAI and
  Anthropic official-SDK adapters, selected by `LLM_PROVIDER`
- Tool-calling loop with a `get_fx_rate(base, quote)` tool (NBP API, Frankfurter
  fallback, timeouts, structured errors)
- Structured output `Answer{answer, citations, confidence}` validated with Pydantic
- Settings via `pydantic-settings`, `.env` / `.env.example`
- Day 2 ingestion: pypdf → structure-aware chunker (512–800 tokens, ~12%
  overlap, section metadata, content hash) → embeddings → Postgres + pgvector
  with an HNSW index (`vector_cosine_ops`, `m=16`, `ef_construction=64`)
- Default embedder is the local multilingual model
  `intfloat/multilingual-e5-base` (e5 `query:` / `passage:` prefixes,
  L2-normalised). `BAAI/bge-m3` is an option; OpenAI
  `text-embedding-3-small` is used only when `OPENAI_API_KEY` is set.
- OpenTelemetry traces for `/chat`, `/ingest` and `/search`: FastAPI, psycopg
  and httpx auto-instrumentation, plus manual GenAI spans (`invoke_agent`,
  `chat`, `execute_tool`, `embeddings`, `retrieval`). Off until you set
  `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_TRACES_EXPORTER=console`. See
  [docs/otel-notes.md](docs/otel-notes.md).
- `docker-compose` (api + Postgres/pgvector; `otel` profile for Collector + Jaeger;
  Langfuse stub commented for Day 6)
- Multi-stage, non-root `Dockerfile` built with `uv`
- pre-commit (ruff, mypy), pytest, GitHub Actions CI (no real API calls;
  integration tests that need Postgres are skipped unless `DATABASE_URL` is set)
- `make smoke` — optional live call to xAI when `XAI_API_KEY` is set
- `make db-init`, `make ingest`, `make bench` — schema, corpus ingest, HNSW vs
  exact-search numbers in [docs/retrieval-notes.md](docs/retrieval-notes.md)

## Planned architecture

```mermaid
flowchart TB
    UI["React chat UI<br/>streaming + citations"]
    CI["GitLab CI<br/>lint → unit → eval gate → build → Trivy → helm"]
    GW["FastAPI gateway<br/>authN · tenant_id · rate limit · audit"]
    AG["LangGraph agent<br/>router → retrieve → grade → answer → guardrail<br/>checkpoints, HITL interrupt"]

    UI --> GW --> AG
    CI -.-> GW

    AG --> MCP["MCP client / servers<br/>fin-tools: FX, bond calc, SQL"]
    AG --> RET["Retriever service<br/>hybrid: pgvector HNSW + BM25/Qdrant<br/>+ cross-encoder rerank"]
    AG --> LLM["LLM gateway OpenAI-compatible"]
    AG --> GRD["Guardrails<br/>NeMo / input+output checks"]

    LLM --> VLLM["vLLM GPU / Ollama laptop"]
    LLM --> NIM["NVIDIA NIM / Triton + TRT-LLM"]
    LLM --> CLOUD["Cloud fallback<br/>xAI / OpenAI / Anthropic / Bedrock"]

    RET --> ING["Ingestion<br/>Docling / VLM PDF → chunks → embeddings<br/>bge-m3 / e5"]
    ING --> PG[("Postgres + pgvector")]
    RET --> QD[("Qdrant sparse/BM25")]

    OBS["Observability<br/>OpenTelemetry → Collector<br/>Jaeger / Langfuse / Grafana"]
    DEP["Deploy<br/>compose on a laptop → Helm on kind/k3s<br/>→ rented GPU node + GPU Operator"]
    GW -.-> OBS
    AG -.-> OBS
    DEP -.-> GW
```

Days 1–2 are the **LLM gateway** plus the **ingestion / pgvector** slice.
The weekend OpenTelemetry lab adds vendor-neutral traces. Hybrid search, MCP,
Langfuse (as an OTLP backend), evals, local inference and Kubernetes land on
later days.

## Quickstart

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). An
[xAI API key](https://console.x.ai/) for live calls.

```bash
git clone https://github.com/saequus/agentic-rag-platform.git
cd agentic-rag-platform
uv sync --dev
cp .env.example .env
# edit .env and set XAI_API_KEY=...

uv run fastapi dev src/agentic_rag_platform/main.py --port 8000
```

Health check and a chat turn:

```bash
curl -s http://127.0.0.1:8000/health | jq

curl -s http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"What is the mid rate of 1 USD in PLN today? Cite your source."}' | jq
```

Example response shape:

```json
{
  "answer": "1 USD is 4.25 PLN at the NBP mid rate published on 2026-11-19.",
  "citations": ["https://api.nbp.pl/api/exchangerates/rates/a/usd/?format=json"],
  "confidence": 0.93,
  "provider": "xai",
  "model": "grok-4.7",
  "tool_calls": [
    {
      "id": "call_1",
      "name": "get_fx_rate",
      "arguments": {"base": "USD", "quote": "PLN"},
      "result": "{\"base\": \"USD\", \"quote\": \"PLN\", \"rate\": 4.25, \"source\": \"NBP\"}"
    }
  ]
}
```

### Postman

A Collection v2.1 and a `local` environment live in [`postman/`](postman/). In the Postman desktop app: **Import** → choose the files `postman/agentic-rag-platform.postman_collection.json` and `postman/local.postman_environment.json`, then select the **local** environment in the top-right corner (`baseUrl=http://localhost:8000`). Interactive docs are at [http://localhost:8000/docs](http://localhost:8000/docs) (`/openapi.json` for the schema). Do not send **Ingest corpus** unless you want a full re-embed; use **Ingest a single PDF** for a quick check.

### Docker Compose

```bash
cp .env.example .env   # set XAI_API_KEY
docker compose up --build
curl -s http://127.0.0.1:8000/health
```

Postgres with pgvector is included for ingestion. After the first
`docker compose up`, run `DATABASE_URL=postgresql://rag:rag@localhost:5432/rag make db-init`
against the published port (or rely on a local Postgres.app database — see
below). A Langfuse service is left commented in `docker-compose.yml` (Day 6).
OpenTelemetry Collector + Jaeger are available with `--profile otel`
(see [docs/otel-notes.md](docs/otel-notes.md)).

### OpenTelemetry (optional)

```bash
make otel-up
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
uv run fastapi dev src/agentic_rag_platform/main.py --port 8000
# send a /chat or /search request, then open http://127.0.0.1:16686
```

Without Docker, `OTEL_TRACES_EXPORTER=console` prints spans to stdout.
Details and the GenAI attribute map: [docs/otel-notes.md](docs/otel-notes.md).

### Tests and live smoke

```bash
make lint          # ruff + mypy
make test          # unit tests, all HTTP/LLM calls mocked
make smoke         # real xAI call; requires XAI_API_KEY; not run in CI
```

Integration tests for pgvector run only when `DATABASE_URL` is set
(`pytest -m integration`). CI does not set it, so those tests are skipped.

### Day 2: ingest and search (local Postgres.app)

The default `DATABASE_URL` in `.env.example` is
`postgresql://slava@localhost:5432/agentic_rag` — the database already
created on the development Mac (Postgres.app 18, trust auth, pgvector
0.8.7 enabled). From a fresh clone:

```bash
git pull
uv sync --dev
cp .env.example .env          # already points at agentic_rag; add XAI_API_KEY if you want /chat
make db-init                  # extension, tables, HNSW index
make ingest                   # downloads 23 public PDFs into data/raw (gitignored), then embeds
make bench                    # 50 queries, HNSW vs exact search, updates docs/retrieval-notes.md
```

The first `make ingest` also downloads `intfloat/multilingual-e5-base`
(~1 GB) from Hugging Face. To compare with OpenAI embeddings, set
`EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY`, then re-run
`make db-init` against an empty database (the vector dimension is 1536
instead of 768) and `make ingest` again.

Search via the API (after `make dev` or `uv run fastapi dev ...`):

```bash
curl -s http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"ECB interest rate decision and the 2% inflation target","k":5}' | jq
```

The committed corpus is 12 ECB Economic Bulletins and 11 official Polish
retail Treasury bond issue letters. NBP inflation reports are documented
in [docs/retrieval-notes.md](docs/retrieval-notes.md) but are not in
`data/urls.txt`, because nbp.pl served a bot challenge from the network
used to verify the URLs.

CI on `main` and on pull requests runs ruff, mypy and pytest only. No secrets
are required and no provider APIs are contacted.

## Switching providers

```bash
# default
LLM_PROVIDER=xai
XAI_API_KEY=...
XAI_MODEL=grok-4.7          # optional; this is the default
XAI_BASE_URL=https://api.x.ai/v1

# alternatives, for learning their APIs
LLM_PROVIDER=openai         OPENAI_API_KEY=...     OPENAI_MODEL=gpt-4o
LLM_PROVIDER=anthropic      ANTHROPIC_API_KEY=...  ANTHROPIC_MODEL=claude-sonnet-5
```

`XAI_BASE_URL` is the documented OpenAI-compatible REST base
(`https://api.x.ai/v1`). The official SDK used by `XAIProvider` talks gRPC to
`api.x.ai`. See the [quickstart](https://docs.x.ai/developers/quickstart).

## Repository layout

```
src/agentic_rag_platform/
  main.py                 FastAPI app: /health, /chat, /ingest, /search
  settings.py             pydantic-settings
  schemas.py              Answer, ChatRequest, Message, ToolCall
  providers/              xAI (default), OpenAI, Anthropic + factory
  tools/                  get_fx_rate + registry
  agent/loop.py           tool loop + structured final answer
  ingest/                 Day 2: PDF load, chunker, embedders, pgvector store
migrations/001_init.sql   documents + chunks + HNSW index
data/urls.txt             verified public PDF list (files themselves gitignored)
tests/                    mocked unit tests + optional pgvector integration tests
scripts/smoke.py          optional live xAI check
scripts/download_corpus.py
scripts/bench.py          HNSW vs exact search → docs/retrieval-notes.md
docs/learning-notes.md    vendor API comparison with doc links
docs/retrieval-notes.md   chunking, embeddings, xAI embeddings check, bench numbers
postman/                  Collection v2.1 + local environment for the API
```

## Roadmap (20 weekdays, 19 Nov – 16 Dec 2026)

- [x] **Day 1** — LLM gateway, provider interface, tool calling, structured output, CI
- [x] **Day 2** — Ingestion, chunking, embeddings, pgvector
- [ ] **Day 3** — Hybrid search, Qdrant, reranking
- [ ] **Day 4** — LangGraph agents (LangChain / LlamaIndex / Pydantic AI trade-offs)
- [ ] **Day 5** — MCP server and client
- [ ] **Day 6** — Observability: Langfuse, prompt versioning, OpenTelemetry
- [ ] **Day 7** — Evals: Ragas, DeepEval, golden set, eval gate in CI
- [ ] **Day 8** — vLLM and Ollama local inference
- [ ] **Day 9** — SLMs, quantization, PEFT/LoRA basics
- [ ] **Day 10** — NVIDIA stack I: GPUs, CUDA, TensorRT-LLM, Triton, NIM
- [ ] **Day 11** — NVIDIA stack II: NeMo Guardrails, AI Enterprise, Blueprints
- [ ] **Day 12** — VLMs and multimodal document parsing
- [ ] **Day 13** — Kubernetes for AI: Helm, GPU scheduling, autoscaling
- [ ] **Day 14** — CI/CD and LLMOps with GitLab CI (and Jenkins)
- [ ] **Day 15** — AI security: prompt injection, OWASP LLM Top 10, access control
- [ ] **Day 16** — Hybrid cloud / on-prem: Bedrock, Vertex, Azure, Terraform, React UI
- [ ] **Day 17** — Documentation, system diagrams, solution design, RFP/RFI
- [ ] **Days 18–20** — Mock interviews (HR, system design, pre-sales)

## License

MIT. See [LICENSE](LICENSE).
