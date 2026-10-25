# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions
may break).

## [Unreleased]

## [0.1.0] - 2026-10-25

First release.

### Security
- MCP servers no longer inherit Flux's environment, so they cannot read its
  LLM API keys or database URLs. Each gets a fixed set of non-secret
  variables (`PATH`, `HOME`, locale, XDG dirs, CA-bundle paths; see README
  "MCP tool servers"), plus the variables named in its new
  `tools.mcp[].env` list.
- An agent can no longer call a tool outside its `tools:` list; such a call
  returns `{"error": "tool '<name>' is not available"}` to the model.
- `POST /runs/{id}/approve` only accepts a stage the run is paused on; any
  other `stage_id` returns 409 and leaves the gate in place.
- Passwords in Redis/Postgres URLs are redacted from logs and error messages.

### Added
- `join: any` on a stage: it runs after whichever of its dependencies ran,
  so conditional branches can rejoin. `join: all` (the default) keeps the
  previous behaviour.
- `POST /runs/{run_id}/cancel` stops a run, paused or running, with new run
  statuses `cancelling` and `cancelled` and a terminal `WorkflowCancelled`
  event. Library users get `run_workflow_cancellable`.
- `Idempotency-Key` header on `POST /runs` and `POST /chat`: a retried
  request returns the original run instead of starting a second one; reusing
  a key with a different body is `422`.
- `llm.providers` in settings: name extra LLM providers, including any
  OpenAI-compatible server (Ollama, vLLM, ...) by `base_url`. `anthropic`,
  `openai` and `mock` stay built in. Library users can implement `LlmBackend`
  and `LlmRegistry::register` a custom provider.
- `refine` on an agent stage: an evaluator-optimizer loop. An evaluator agent
  judges the stage's answer and the stage retries with its feedback, up to
  `max_rounds`, then fails or (`on_exhausted: accept_last`) keeps the last
  answer. New `WorkflowEvent::RefineRound` reports each round. See README
  "Refining a stage's answer" and `workflows/refine-mock.yaml`.

### Changed (breaking)
- An MCP server that read a token from Flux's environment (for example
  `GITHUB_PERSONAL_ACCESS_TOKEN`) must now list it in `tools.mcp[].env`.
  `flux_tool_client::mcp::McpServer` gains an `env` field.
- New `WorkflowEvent::RefineRound` (see Added). `WorkflowEvent` is not
  `#[non_exhaustive]`, so exhaustive matches downstream stop compiling. It is
  also a stored-data change: an older server cannot read a run log that
  contains it, so do not run mixed versions against one event store.
- Stage ids may no longer be `candidate`, `verdict`, `feedback`, `input` or
  `round` (reserved for refine templates), in addition to `workflow` and
  `ctx`.
- Run cancellation (see Added) changes library APIs: `flux_runner::run_agent`
  takes a `CancellationToken`; `RunOutcome` gains `Cancelled`, `RunStatus`
  gains `Cancelling` and `Cancelled`, and `WorkflowEvent` gains
  `WorkflowCancelled`. None is `#[non_exhaustive]`, so exhaustive matches
  downstream stop compiling.
- `EventStore` gains the required methods `claim_idempotency_key` and
  `release_idempotency_key` (for the `Idempotency-Key` header), so custom
  stores must implement them.
- LLM calls go through the new `flux-llm` layer instead of an embedded
  TensorZero gateway. `run_agent`, `run_workflow` and
  `run_workflow_cancellable` take an `&LlmRegistry`.
- `flux_config::Provider` is removed; `ModelConfig.provider` is a
  provider-name `String`.
- New `AgentEvent::Usage` and `AgentRun.usage` report token usage.
  `AgentEvent` is not `#[non_exhaustive]`, so exhaustive matches downstream
  stop compiling, and clients of the SSE stream or stored run events see a
  new `Usage` event kind. Usage counts only LLM calls that complete: failed
  attempts that are retried, and calls that fail after streaming text,
  report none.
- `flux_runner::CapturedToolCall` gains `invalid_args` and `Turn` gains
  `opaque` and `usage`. Neither is `#[non_exhaustive]`, so struct literals
  downstream stop compiling.
- When `max_tokens` is unset, Anthropic requests send 4096, or the thinking
  budget plus 4096 (TensorZero chose a per-model default). Agents loaded
  through Flux binaries already get `agent_defaults.max_tokens`, so only
  library callers building `ModelConfig` by hand are affected.
- An `llm.providers` entry named `anthropic` or `openai` replaces the
  built-in but inherits its `api_key_env` unless it sets its own.
- `flux-server` and `run_workflow` fail at boot if any loaded agent names an
  LLM provider that is not configured (previously only missing API keys were
  warned about).
- LLM requests are no longer retried on 4xx responses other than 408, 409
  and 429.
- Malformed tool arguments from the model are returned to it as an error
  instead of being passed on as `null`.
- An OpenAI-compatible provider with an `api_key_env` requires that key: a
  missing key fails the request with "<VAR> is not set" instead of OpenAI's
  raw 401 (also reported by the boot preflight). A provider without
  `api_key_env` sends no `Authorization` header (what local servers expect).
- The assistant's tool-call turn is recorded before its tools run, so a run
  cancelled mid-tool-call keeps it in conversation memory; the unanswered
  call is dropped when memory is read back.
- Stage ids must be identifiers (letters, digits, `_`; not starting with a
  digit) and may not be `workflow` or `ctx`.
- A stage's `input`, `when` or `args` may only read stages in its
  `depends_on` (directly or transitively); this is checked when the workflow
  loads. Template syntax errors are now also reported at load time.

### Removed
- TensorZero, and with it the `tensorzero` git dependency and its
  `deny.toml` exceptions.

### Fixed
- A run that panics is recorded as `failed` and ends its event stream with a
  `WorkflowFailed` event, instead of reading as `running` until it expires.
- A `/chat` run keeps running when the client disconnects, instead of being
  dropped mid-stage with its status stuck at `running`. Stop it with the
  cancel endpoint.
- A remembering stage whose model call or output check fails still saves the
  user's message to conversation memory.
- gRPC tool calls have a 30s connect timeout and a 120s call deadline instead
  of none.
- An LLM attempt fails after 180s of provider silence instead of hanging.
- SSE streams send keep-alive comments, so proxies no longer cut a stream
  while its run waits at an approval gate, and a stream ends once its run's
  state has expired instead of polling until the client disconnects.
- Every event a run appends refreshes the TTL of all its Redis keys, so a run
  longer than `redis.ttl_secs` keeps its status, and an approval gets a full
  TTL from the moment the run pauses. Run keys now carry a `{run_id}` hash
  tag (Redis Cluster-safe); runs stored under the old key names are not read.

[Unreleased]: https://github.com/thinkbytelabs/flux-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thinkbytelabs/flux-ai/releases/tag/v0.1.0
