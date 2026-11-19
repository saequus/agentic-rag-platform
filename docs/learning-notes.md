# Tool calling and structured outputs: xAI vs OpenAI vs Anthropic

Day 1 of this repo implements one provider-agnostic interface
(`LLMProvider.complete(messages, tools=, response_schema=)`) and three official
SDK adapters. The loop in `src/agentic_rag_platform/agent/loop.py` is the same
for every vendor: run tools until the model stops calling them, then request a
Pydantic `Answer{answer, citations, confidence}`.

The notes below are the differences you should be able to explain in an
interview. They are based on the vendor docs as of November 2026, not on
hearsay.

## Why a thin internal interface

A good platform does not leak vendor message formats into the agent. The
gateway owns retries, timeouts, budgets, logging and schema validation. Each
adapter maps:

- a generic `Message` (`system` / `user` / `assistant` / `tool`)
- a generic `ToolSpec` (name, description, JSON Schema parameters)
- an optional Pydantic `response_schema`

onto that vendor's SDK. Switching `LLM_PROVIDER` (or, later, routing per tenant)
should not change the rest of the system. This is also how you put vLLM or
NVIDIA NIM behind the same interface: they expose an OpenAI-compatible API.

## xAI (default in this repo)

- Docs: [Quickstart](https://docs.x.ai/developers/quickstart),
  [Function calling](https://docs.x.ai/developers/tools/function-calling),
  [Structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs).
- Official Python SDK: [`xai-sdk`](https://pypi.org/project/xai-sdk/) (this
  repo uses 1.21.x). `AsyncClient(api_key=..., api_host="api.x.ai")` speaks
  **gRPC**. The same account also has an **OpenAI-compatible REST** surface at
  `https://api.x.ai/v1` (Responses API: `OpenAI(base_url=...).responses.create`).
- Default model here: `grok-4.7` (current flagship for chat and agents;
  override with `XAI_MODEL`). See [Models](https://docs.x.ai/docs/models).
- **Tool calling:** define tools with `tool(name, description, parameters)`
  where `parameters` is a JSON Schema object. After `await chat.sample()`, read
  `response.tool_calls`. Append the raw response, then
  `tool_result(result, tool_call_id=call.id)`, then sample again.
- Tool arguments **always** conform to the tool schema. The `strict` flag is
  implicitly always `true` — you do not opt in.
- **Structured outputs:** `response, parsed = await chat.parse(Answer)` (or
  pass `response_format=Answer` into `chat.create`). This is
  `response_format.type = "json_schema"`. `"json_object"` is also available
  when you only need well-formed JSON. Grok 4 family models can combine tools
  with a structured final answer.

## OpenAI

- Docs: [Function calling](https://platform.openai.com/docs/guides/function-calling),
  [Structured outputs](https://platform.openai.com/docs/guides/structured-outputs).
- Official SDK: `openai`. This adapter uses Chat Completions because that is
  still the API most interviewers walk through. The newer Responses API
  (`client.responses.parse`) does the same job on a different surface.
- **Tool calling:** tools are wrapped as
  `{"type": "function", "function": {name, description, parameters, strict}}`.
  The assistant message contains `tool_calls`; you append it, then append
  `role="tool"` messages with `tool_call_id`.
- `strict` is **opt-in** (default `false`). This adapter sets `strict=True` so
  behaviour is closer to xAI.
- **Structured outputs:** `client.chat.completions.parse(..., response_format=Answer)`
  sends `response_format = {type: "json_schema", json_schema: {name, strict, schema}}`
  and returns `message.parsed`.
- Three levels, in increasing strength:
  1. Prompting "reply as JSON" — no guarantee.
  2. JSON mode (`{"type": "json_object"}`) — valid JSON, no schema.
  3. Structured outputs (`json_schema` + `strict`) — schema-constrained.

## Anthropic

- Docs: [Tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
  [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use),
  [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
- Official SDK: `anthropic`. Default model here: `claude-sonnet-5` (override
  with `ANTHROPIC_MODEL`).
- **Tool calling:** the schema is flatter than OpenAI's:
  `{"name", "description", "input_schema"}` — no `type: function` wrapper, and
  parameters are called `input_schema`. The model returns
  `stop_reason="tool_use"` and `tool_use` content blocks (`id`, `name`,
  `input` already a dict). You echo the assistant `content`, then send results
  as a **user** message with `tool_result` blocks (`tool_use_id`). There is
  **no** `role="tool"`.
- The system prompt is a top-level `system=` argument, not a message in the
  list. This adapter splits `Message(role="system")` out accordingly.
- `strict: true` is opt-in for tool inputs. This adapter turns it on.
- **Structured outputs:** `client.messages.parse(..., output_format=Answer)`
  is the recommended helper. The SDK sends `output_config.format` as a JSON
  Schema and returns `parsed_output`. JSON outputs (what Claude *says*) and
  strict tool use (how it *calls* your functions) can be combined.

## Side-by-side

| Topic | xAI (`xai-sdk`) | OpenAI (Chat Completions) | Anthropic (Messages) |
| --- | --- | --- | --- |
| Tool object | `tool(name, description, parameters)` | `{type: "function", function: {…}}` | `{name, description, input_schema}` |
| Tool result role | `ROLE_TOOL` / `tool_result(...)` | `role="tool"` + `tool_call_id` | `role="user"` + `tool_result` / `tool_use_id` |
| Arguments shape | JSON string (`function.arguments`) | JSON string (`function.arguments`) | Already a dict (`block.input`) |
| Strict tool args | Always on | Opt-in `strict: true` | Opt-in `strict: true` |
| Structured helper | `chat.parse(Model)` | `chat.completions.parse(..., response_format=Model)` | `messages.parse(..., output_format=Model)` |
| Schema field name | `response_format` (json_schema) | `response_format.json_schema` | `output_config.format` |
| System prompt | `system()` message | `role="system"` message | Top-level `system=` |
| Transport (this repo) | gRPC to `api.x.ai` (REST also at `/v1`) | HTTPS | HTTPS |

## What we still validate ourselves

Constrained decoding is not a substitute for server-side validation. Every
final answer goes through `Answer.model_validate` (confidence must be in
`[0, 1]`, the text must be non-empty). Tool arguments go through `FxRateArgs`
before the HTTP call. If a provider returns free text instead of schema JSON,
the loop wraps it in an `Answer` with a low confidence rather than crashing.
