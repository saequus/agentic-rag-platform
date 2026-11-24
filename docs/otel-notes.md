# OpenTelemetry notes (weekend lab)

HTTP server spans come from FastAPI 0.165 native OpenTelemetry (with
`/health` and docs excluded). The gateway, agent loop, tool calls, embeddings and retrieval also emit
[OpenTelemetry](https://opentelemetry.io/docs/concepts/signals/traces/) traces
when an exporter is configured. GenAI attributes follow the official
conventions, which are **Development** status and now live in a dedicated
repository (they moved out of the core semantic-conventions tree in June 2026):

- https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md
- Inference spans: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md
- Tool spans: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md
- Retrieval spans: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/retrieval.md
- Agent spans: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md

## What a `/chat` trace looks like

```
POST /chat                         (FastAPI auto-instrumentation, span kind SERVER)
└─ invoke_agent finance_assistant  (gen_ai.operation.name=invoke_agent)
   ├─ chat grok-4.7                (CLIENT; provider, model, temperature)
   ├─ execute_tool get_fx_rate     (tool name + call id; no arguments/results)
   │  └─ HTTP GET api.nbp.pl       (httpx auto-instrumentation, if used)
   └─ chat grok-4.7                (structured-output turn, gen_ai.output.type=json)
```

A `/search` trace:

```
POST /search
└─ retrieval                       (top_k, tenant.id, data_source.id=pgvector)
   ├─ embeddings <model>           (dimensions, input count; no query text)
   └─ PostgreSQL SELECT            (psycopg auto-instrumentation)
```

`GET /health`, `/docs` and `/openapi.json` are excluded.

Prompts, tool arguments, tool results and retrieved chunk text are **not**
attached to spans by default (`OTEL_REDACT_CONTENT=true`). Set it to `false`
only on a local throwaway backend if you want `gen_ai.retrieval.query.text`
for debugging.

Token usage (`gen_ai.usage.input_tokens` / `output_tokens`) is recorded when
a provider fills `ProviderResponse.input_tokens` / `output_tokens`. The xAI,
OpenAI and Anthropic adapters do not populate those fields yet — a small
follow-up.

## Run it locally

Tracing stays off until you set an exporter. Two easy options:

### 1. Collector + Jaeger (Docker)

```bash
make otel-up          # otel/opentelemetry-collector-contrib + Jaeger UI
# in another terminal, from the repo root:
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_SERVICE_NAME=agentic-rag-platform
uv run fastapi dev src/agentic_rag_platform/main.py --port 8000
```

Open [http://127.0.0.1:16686](http://127.0.0.1:16686), pick the
`agentic-rag-platform` service, and send `POST /chat` or `POST /search`.
The Collector config is `deploy/otel/collector.yaml` (OTLP receiver →
attribute redaction → batch → Jaeger). Tail sampling is sketched in comments.

To run the API in Compose too:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 docker compose --profile otel up --build
```

### 2. Console exporter (no Docker)

```bash
export OTEL_TRACES_EXPORTER=console
uv run fastapi dev src/agentic_rag_platform/main.py --port 8000
```

Spans print to stdout. Useful when Docker is not available.

### 3. Arize Phoenix (optional)

Phoenix is an OTLP backend with a GenAI-oriented UI (OpenInference). After
`phoenix serve`, point `OTEL_EXPORTER_OTLP_ENDPOINT` at the OTLP port shown
in its startup log (often `http://127.0.0.1:6006` or `4317`). Confirm the
port rather than assuming it.

## Configuration

Standard OpenTelemetry environment variables, plus two app-specific ones:

| Variable | Role |
| --- | --- |
| `OTEL_SDK_DISABLED` | Turn the SDK off entirely |
| `OTEL_SERVICE_NAME` | Resource `service.name` (default `agentic-rag-platform`) |
| `OTEL_ENVIRONMENT` | Resource `deployment.environment` (default `local`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP base URL, e.g. `http://127.0.0.1:4318` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` (default) |
| `OTEL_TRACES_EXPORTER` | `otlp`, `console`, or `otlp,console` |
| `OTEL_TRACES_SAMPLER` | default `parentbased_traceidratio` |
| `OTEL_TRACES_SAMPLER_ARG` | ratio 0.0–1.0 (default 1.0, i.e. everything) |
| `OTEL_REDACT_CONTENT` | default `true`; see above |

## How this relates to Langfuse / Phoenix / OpenLLMetry

OpenTelemetry is the **vendor-neutral instrumentation and export standard**,
not a backend. Langfuse (Day 6) can *receive* these traces on its OTLP
endpoint (`/api/public/otel`) and also offers an OTel-native SDK. OpenLLMetry
(Traceloop) and OpenInference (Arize) are instrumentation layers on top of
OTel that add LLM-specific spans; they can export to Langfuse, Phoenix, or
any OTLP collector. For an on-prem Lenovo-style customer, emit OTel from the
platform and let the customer choose the backend.

Day 6 will wire Langfuse as one such backend and add prompt versioning.

## Day 4: LangGraph agent spans

```
POST /chat                                   (FastAPI server span)
└─ invoke_agent finance_assistant            gen_ai.operation.name=invoke_agent, gen_ai.conversation.id=<thread_id>,
   │                                         langgraph.graph=rag_agent, agent.llm_calls, agent.steps
   ├─ node route        └─ chat <model>      (structured output: RouteDecision)
   ├─ node retrieve     └─ retrieval ─ embeddings / rerank / psycopg
   ├─ node grade_docs   └─ chat <model>      (DocGrades)
   ├─ node generate     └─ chat <model>      gen_ai.request.stream=true
   ├─ node check_grounded └─ chat <model>    (GroundingVerdict)
   ├─ node tool_call    └─ execute_tool get_fx_rate ─ httpx
   ├─ node approval     langgraph.interrupted=true (status not ERROR: interrupt is control flow)
   └─ node finalize
```

Node spans carry only metadata (`langgraph.node`, `agent.route`,
`agent.rewrites`, `agent.docs.count`, `agent.relevant.count`, `agent.grounded`,
`agent.fallback_reason`). Questions, passages and answers are never recorded,
whatever `OTEL_REDACT_CONTENT` says. A test checks that a PESEL-like string in
the question doesn't appear on any span.
