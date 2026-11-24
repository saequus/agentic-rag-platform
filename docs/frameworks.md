# Agent frameworks: LangGraph vs Pydantic AI vs a plain loop (Day 4)

This repo has the same retrieve → answer job implemented three ways:

| | Plain loop (`agent/loop.py`, Day 1) | LangGraph (`graph/`, Day 4) | Pydantic AI (`agent/pydantic_ai_agent.py`) |
|---|---|---|---|
| Control flow | A `for` loop: the model calls tools until it answers | Explicit state machine: nodes, conditional edges, cycles | The model decides which tool to call; the framework runs the loop |
| State | Local variables (the message list) | Typed `AgentState`, reducers (`messages` appends) | `deps` (typed, injected) plus message history you pass in |
| Persistence / resume | None | Checkpointer per `thread_id` (Postgres); resume after a crash or a restart | Message history is serialisable; no built-in graph checkpoints |
| Human in the loop | Would have to be built by hand | `interrupt()` + `Command(resume=...)` | Deferred tools / approval exist, but you build the resume flow |
| Streaming | Not implemented | `stream_mode=["updates", "custom"]`: node events + tokens | `agent.run_stream()` for text and structured output |
| Typed output | `Answer` validated with Pydantic | Structured output per node (router, grader, grounding judge) | `output_type=CitedAnswer`: validation and retries built in |
| Loop control | `max_iterations` | `recursion_limit`, rewrite cap, budget in state, deterministic edges | `UsageLimits` (requests and tokens), retries |
| Lines of code here | ~200 | ~700 (nodes, builder, runner, routes) | ~50 of logic |
| Observability | Manual GenAI spans | Manual spans per node (here); LangSmith or OTel exporters | Built-in OTel via Logfire / `instrument=True` |

**When to use which**

- **Plain loop**: one model, a few tools, no need to resume. It's the easiest to
  read and test. You end up rebuilding persistence, HITL and streaming the moment
  you need them.
- **Pydantic AI**: typed dependencies, typed outputs and good developer
  experience. It's the best choice when the model can drive the control flow
  (tool calling) and a service only needs "ask → validated answer". Swapping
  models is a one-line change (`XaiModel` here; OpenAI, Anthropic and others ship too). In tests,
  `TestModel` / `FunctionModel` replace the LLM. Our test uses `TestModel`, so
  there is no network.
- **LangGraph**: when the flow itself is the product. Examples are grading and
  rewrite loops, deterministic exits, approvals that wait hours, durable
  multi-turn threads, subgraphs for multi-agent, and replaying or forking from a
  checkpoint (time travel). The cost is more code and more concepts (reducers,
  checkpoints, the fact that a node re-runs on resume).

**Where LlamaIndex fits.** LlamaIndex is data-centric. Its strength is
ingestion and query: readers / LlamaParse for messy PDFs, node parsers
(chunking), index abstractions over many vector stores, and query engines (sub-question,
router, recursive retrieval) and response synthesizers. In this platform it
would replace parts of `ingest/` and `retrieval/`, not the orchestration. It
also has an agent and workflow layer (`Workflow`, event-driven steps), but for
an explicit, checkpointed agent graph I would still put LangGraph on top of a
LlamaIndex retriever or query engine. The retriever is behind an interface
(`RetrieveFn`), so either one can feed the graph.

**CrewAI / AutoGen / NVIDIA NeMo Agent Toolkit.** CrewAI and AutoGen are
role-based multi-agent abstractions ("researcher", "writer"), fast to prototype
but with less control over each step. NVIDIA's NeMo Agent Toolkit is a
framework-agnostic layer: it wraps LangGraph, LlamaIndex or CrewAI agents with
profiling, evaluation and observability, and ties into NIM. For a platform,
choose on control, observability hooks, license and team familiarity, and keep
business logic (retrieval, tools, prompts) behind interfaces so the
orchestrator can be swapped.

## Interview notes

### Why LangGraph rather than chains or your own loop?
"The agent is a state machine with checkpoints. Retrieval quality is graded,
bad queries are rewritten at most twice, and answers are checked for grounding
before they leave. Each of those is a node with a deterministic edge, not a
hope that the model behaves. The checkpointer gives me thread memory and
human approval that survives a restart. There is an integration test that
interrupts in one process and resumes in another."

### Loop control and cost (what this repo does)
1. `recursion_limit=25` on every run. `GraphRecursionError` becomes an
   "I don't know" answer, not a 500 error.
2. `AGENT_MAX_REWRITES=2` on the retrieve → grade → rewrite cycle.
3. **Budget in state**: `llm_calls` and `max_llm_calls` are checkpointed with
   the turn. Edges check the budget left before starting the next step
   (`generate` + `check_grounded` need 2 calls).
4. **Loop detection**: a rewritten query that was already tried ends the turn
   (`fallback_reason=repeated_query`).
5. **Timeouts per LLM call** (`asyncio.wait_for` / `asyncio.timeout`).
6. **Deterministic exits** first: no relevant documents → no generation; no
   valid `[n]` citation → not grounded, without paying for the judge.
7. One structured call grades all candidates, rather than one call per document
   (5x fewer calls; the trade-off is a longer prompt).

Not done yet: token-based budgets. The adapters don't fill `input_tokens` yet;
the plan is to switch the budget from calls to tokens or cost once they do.

### Memory: short-term vs long-term
- **Short-term** = thread state in the checkpointer (`thread_id`): the
  `messages` list (with an append reducer) and the current turn's state. The
  last N turns go into prompts so follow-ups resolve ("A w drugim okresie?").
  Retention is a policy question: checkpoints contain user text, so set a TTL
  or delete threads (`adelete_thread`). Never put PII on spans.
- **Long-term** = a store namespaced by `(tenant_id, user_id)` for facts and
  preferences, retrieved semantically (LangGraph `BaseStore` / `PostgresStore`
  with an index). Write to it from an explicit, approved tool, like `save_note`
  here, rather than silently. Per-tenant isolation applies to both kinds:
  this API refuses a `thread_id` that belongs to another tenant.

### Human in the loop: the gotchas
- `interrupt()` raises internally. The checkpoint is saved and the run stops.
  On `Command(resume=...)` the **whole node runs again from the top**, and
  `interrupt()` returns the resume value. So no side effects may happen before
  `interrupt()`. Make the write itself idempotent, or put it in its own node.
- A thread with a pending interrupt rejects a new message (HTTP 409) until
  someone resumes it.
- The approval payload is what the human sees: tool name and arguments, not
  the model's reasoning.

### Streaming
Tokens stream while `generate` runs, and `check_grounded` runs afterwards. So
the UI treats tokens as provisional and renders the `done` event, which may be
the fallback. The alternative is to verify first and then stream, which gives
higher latency but nothing to retract. Choose per use case. In regulated
finance, I'd verify first for external users.

### Single vs multi-agent
Start with a single agent and good tools. This graph is "one agent, several
deterministic steps". Go multi-agent when one prompt or tool set gets too big,
when teams own separate domains (e.g. a bonds agent and an ECB-macro agent),
or when you need different models or permissions per step. A supervisor (a
router node, with subgraphs as workers) is easier to evaluate and trace than a
swarm. The costs are latency, context passing and harder evals. Every handoff
is a place to lose information.

### Making agents repeatable across customers (platform angle)
Make the graph config-driven (prompts, retrieval strategy and backend, budgets,
which tools need approval), backed by a shared tool registry (MCP on Day 5), the
shared retrieval service, an eval suite per template (Day 7) and one Helm chart
(Day 13). This is the same idea as Lenovo's validated AI Library use cases.
