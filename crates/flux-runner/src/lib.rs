#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_config::AgentConfig;
use flux_core::conversation::{Role as ConvRole, Turn as ConvTurn};
use flux_core::error::{FluxError, Result};
use flux_core::event::AgentEvent;
use flux_core::output::AgentOutput;
pub use flux_llm::RetryConfig;
use flux_llm::{
    build_system_prompt, stream_chat, ChatRequest, Content, LlmEvent, LlmRegistry, Message,
    OutputFormat, ProviderBlock, Reasoning, Role, ToolDef, Usage,
};
use flux_tool_client::{validate_args, Manifest, ToolCall as TcCall, ToolTransport};
use futures_util::future::join_all;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A captured tool call within a turn.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
    /// `Some(parse error)` when the model's arguments were not valid JSON;
    /// `args` is then `{}` and the tool is not called.
    pub invalid_args: Option<String>,
}

/// Token usage summed over a run's LLM calls. A side stays `None` until some
/// call reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageTotals {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

impl UsageTotals {
    pub fn add(&mut self, u: &Usage) {
        fn sum(total: &mut Option<u64>, n: Option<u32>) {
            if let Some(n) = n {
                *total = Some(total.unwrap_or(0).saturating_add(u64::from(n)));
            }
        }
        sum(&mut self.input_tokens, u.input_tokens);
        sum(&mut self.output_tokens, u.output_tokens);
    }
}

/// One agent invocation's result: its final output plus the full turn
/// sequence, so the caller can persist tool activity without putting it on
/// the event stream.
#[derive(Debug)]
pub struct AgentRun {
    pub output: AgentOutput,
    pub turns: Vec<ConvTurn>,
    /// `Some(reason)` when the run gave up (iteration budget exhausted, event
    /// receiver gone) instead of producing a final answer. The turns are still
    /// real: a remembering stage persists them before the stage is failed.
    pub aborted: Option<String>,
    /// Token usage of every LLM call made, including on an aborted run.
    pub usage: UsageTotals,
}

/// Empty result used on the early-exit paths (receiver dropped, iteration
/// budget exhausted).
fn empty_run() -> AgentRun {
    AgentRun {
        output: AgentOutput {
            message: String::new(),
            structured: None,
        },
        turns: Vec::new(),
        aborted: Some("event receiver closed".to_string()),
        usage: UsageTotals::default(),
    }
}

/// An `AgentRun` for a run that gave up partway: no output, but the turns
/// accumulated so far so a remembering stage still records what happened.
///
/// Returning no turns instead would drop the user's own message on the floor —
/// the next turn of the conversation would show no trace the question was ever
/// asked, and the model would answer it as if it were new.
///
/// `reason` closes the log with an assistant turn. Without it the log would
/// end on the tool-result turn the loop was in the middle of, putting two
/// user-role messages back to back once the next question is appended. The
/// note is short and factual because it does reach the model on the next
/// turn — which is better context than an unexplained gap.
///
/// A run cancelled during a tool call stores the assistant tool-call turn
/// without a result, followed by the stop note; it relies on recall's
/// `trim_to_exchange_boundary` dropping the unanswered call.
fn aborted_run(mut turns: Vec<ConvTurn>, usage: UsageTotals, reason: &str) -> AgentRun {
    turns.push(ConvTurn::text(
        ConvRole::Assistant,
        format!("[run stopped: {reason}]"),
    ));
    AgentRun {
        output: AgentOutput {
            message: String::new(),
            structured: None,
        },
        turns,
        aborted: Some(reason.to_string()),
        usage,
    }
}

/// Accumulates one inference turn's streamed events.
#[derive(Debug, Default)]
pub struct Turn {
    pub text: String,
    pub tool_calls: Vec<CapturedToolCall>,
    /// Provider blocks (e.g. signed thinking) echoed back within the tool
    /// loop; never written to conversation memory.
    pub opaque: Vec<ProviderBlock>,
    pub usage: Option<Usage>,
}

impl Turn {
    /// Fold one streamed event; return an AgentEvent to forward immediately
    /// (Thought, Text, Usage) or None (tool calls and opaque blocks are
    /// buffered until the turn ends). Text deltas are forwarded AND
    /// accumulated into `self.text` for the final message. A `Usage` event's
    /// `model` is the provider-reported one, or empty when it reported none.
    pub fn on_event(&mut self, event: LlmEvent) -> Option<AgentEvent> {
        match event {
            LlmEvent::Thought(text) => Some(AgentEvent::Thought { text }),
            LlmEvent::Text(t) => {
                self.text.push_str(&t);
                Some(AgentEvent::Text { delta: t })
            }
            LlmEvent::ToolCall { id, name, args } => {
                self.tool_calls.push(CapturedToolCall {
                    id,
                    name,
                    args,
                    invalid_args: None,
                });
                None
            }
            LlmEvent::InvalidToolCall { id, name, error } => {
                self.tool_calls.push(CapturedToolCall {
                    id,
                    name,
                    args: serde_json::json!({}),
                    invalid_args: Some(error),
                });
                None
            }
            LlmEvent::Opaque(block) => {
                self.opaque.push(block);
                None
            }
            LlmEvent::Usage(u) => {
                let event = AgentEvent::Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    model: u.model.clone().unwrap_or_default(),
                };
                self.usage = Some(u);
                Some(event)
            }
            _ => None,
        }
    }
}

/// Build the tool definitions sent to the model.
///
/// For each tool name referenced in `cfg.tools`, finds the matching `ToolSpec` in
/// the manifest. A reference that is absent from the manifest is a configuration
/// error (`FluxError::Config`).
pub fn build_tool_defs(cfg: &AgentConfig, manifest: &Manifest) -> Result<Vec<ToolDef>> {
    cfg.tools
        .iter()
        .map(|name| {
            let spec = manifest.find(name).ok_or_else(|| {
                FluxError::Config(format!(
                    "tool '{name}' is declared in agent config but not found in the manifest"
                ))
            })?;
            Ok(ToolDef {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec.parameters.clone(),
            })
        })
        .collect()
}

/// The request for `cfg`'s model, without messages.
fn base_request(cfg: &AgentConfig, tools: Vec<ToolDef>) -> ChatRequest {
    let model = &cfg.model;
    let reasoning = if model.reasoning_effort.is_some() || model.thinking_budget_tokens.is_some() {
        Some(Reasoning {
            effort: model.reasoning_effort.clone(),
            budget_tokens: model.thinking_budget_tokens,
        })
    } else {
        None
    };
    ChatRequest {
        model: model.profile.clone().unwrap_or_default(),
        system: build_system_prompt(cfg),
        messages: Vec::new(),
        tools,
        output: cfg
            .guardrails
            .provider_output_schema()
            .map(|schema| OutputFormat::Json {
                schema: schema.clone(),
            })
            .unwrap_or(OutputFormat::Text),
        max_tokens: model.max_tokens,
        temperature: model.temperature,
        reasoning,
    }
}

/// A user message containing a single text content block.
fn user_text_message(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![Content::Text(text.to_string())],
    }
}

/// Map one stored conversation turn to a Message, rebuilding any tool
/// call / tool result content blocks it carries.
///
/// `None` for a turn that would yield no content blocks at all — empty text and
/// no tool activity. Providers reject a message with an empty `content` array,
/// and one such row (a model reply that came back empty, or a legacy row
/// written before tool columns existed) would otherwise fail every subsequent
/// turn of that conversation. Skipping it costs one blank line of history.
fn history_message(turn: &ConvTurn) -> Option<Message> {
    let mut content: Vec<Content> = Vec::new();
    if !turn.content.is_empty() {
        content.push(Content::Text(turn.content.clone()));
    }
    for call in &turn.tool_calls {
        content.push(Content::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            args: call.args.clone(),
        });
    }
    for result in &turn.tool_results {
        content.push(Content::ToolResult {
            id: result.id.clone(),
            name: result.name.clone(),
            result: result.result.clone(),
        });
    }
    if content.is_empty() {
        return None;
    }
    let role = match turn.role {
        ConvRole::User => Role::User,
        ConvRole::Assistant => Role::Assistant,
    };
    Some(Message { role, content })
}

/// Build the initial message list: prior conversation history (oldest-first)
/// followed by the current user message.
fn assemble_messages(history: &[ConvTurn], user_message: &str) -> Vec<Message> {
    let mut messages: Vec<Message> = history.iter().filter_map(history_message).collect();
    messages.push(user_text_message(user_message));
    messages
}

/// The assistant message for a tool-calling turn: its opaque provider blocks
/// first (a thinking block must precede the text and tool calls it led to),
/// then its text (if any), then one ToolCall block per captured call.
fn assistant_message(turn: &Turn) -> Message {
    let mut content: Vec<Content> = turn.opaque.iter().cloned().map(Content::Opaque).collect();
    if !turn.text.is_empty() {
        content.push(Content::Text(turn.text.clone()));
    }
    for call in &turn.tool_calls {
        content.push(Content::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            args: call.args.clone(),
        });
    }
    Message {
        role: Role::Assistant,
        content,
    }
}

/// A tool-result message (sent as a User role message).
fn tool_result_message(call: &CapturedToolCall, result: &serde_json::Value) -> Message {
    Message {
        role: Role::User,
        content: vec![Content::ToolResult {
            id: call.id.clone(),
            name: call.name.clone(),
            result: result.clone(),
        }],
    }
}

/// Execute a turn's tool calls concurrently, preserving call order.
///
/// Emits a `ToolCall` event per call up front, runs all calls concurrently
/// (validation + transport), then emits a `ToolResult` per call. Any per-call
/// failure (malformed arguments, a tool not in `allowed`, schema validation,
/// or transport) becomes a `{"error": <msg>}` value
/// fed back to the model rather than aborting the run. Returns one value per
/// call, in input order. Infallible.
async fn run_tool_calls<T: ToolTransport>(
    transport: &T,
    manifest: &Manifest,
    allowed: &[String],
    context: Option<&serde_json::Value>,
    calls: &[CapturedToolCall],
    tx: &mpsc::Sender<AgentEvent>,
) -> Vec<serde_json::Value> {
    // Announce all calls first (in order).
    for call in calls {
        let _ = tx
            .send(AgentEvent::ToolCall {
                tool: call.name.clone(),
                args: call.args.clone(),
            })
            .await;
    }

    // Run every call concurrently; each future yields a result Value.
    let futures = calls.iter().map(|call| async move {
        if let Some(error) = &call.invalid_args {
            tracing::warn!(tool = %call.name, "model sent malformed tool arguments");
            flux_core::metrics::record_tool_call("error");
            return serde_json::json!({
                "error": format!("invalid tool arguments: {error}")
            });
        }
        // The transport serves every backend's tools, not just this agent's:
        // a model-named tool outside the agent's declared list must not run.
        if !allowed.contains(&call.name) {
            tracing::warn!(tool = %call.name, "model called an undeclared tool");
            flux_core::metrics::record_tool_call("error");
            return serde_json::json!({
                "error": format!("tool '{}' is not available", call.name)
            });
        }
        if let Some(spec) = manifest.find(&call.name) {
            if let Err(e) = validate_args(spec, &call.args) {
                tracing::warn!(tool = %call.name, "tool call failed");
                flux_core::metrics::record_tool_call("error");
                return serde_json::json!({ "error": e.to_string() });
            }
        }
        match transport
            .call_tool(&TcCall {
                name: call.name.clone(),
                args: call.args.clone(),
                context: context.cloned(),
            })
            .await
        {
            // Prefer a transport's compact model-facing projection when it set
            // one: this value is serialized into the prompt again on every
            // later turn. A tool may legitimately return an `error` field in a
            // successful payload — success/failure is decided by the transport
            // result, never by probing the payload for a key.
            Ok(r) => {
                tracing::debug!(tool = %call.name, "tool call ok");
                flux_core::metrics::record_tool_call("ok");
                r.for_model.unwrap_or(r.result)
            }
            Err(e) => {
                tracing::warn!(tool = %call.name, "tool call failed");
                flux_core::metrics::record_tool_call("error");
                serde_json::json!({ "error": e.to_string() })
            }
        }
    });
    let results: Vec<serde_json::Value> = join_all(futures).await;

    // Emit one ToolResult per call, in order.
    for (call, result) in calls.iter().zip(&results) {
        let _ = tx
            .send(AgentEvent::ToolResult {
                tool: call.name.clone(),
                result: result.clone(),
            })
            .await;
    }

    results
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Fold one LLM event into `turn` and `usage`; return the AgentEvent to
/// forward, if any. A `Usage` the provider did not attribute to a model is
/// attributed to the agent's profile.
fn fold_event(
    turn: &mut Turn,
    usage: &mut UsageTotals,
    cfg: &AgentConfig,
    event: LlmEvent,
) -> Option<AgentEvent> {
    if let LlmEvent::Usage(u) = &event {
        usage.add(u);
    }
    match turn.on_event(event)? {
        AgentEvent::Usage {
            input_tokens,
            output_tokens,
            model,
        } if model.is_empty() => Some(AgentEvent::Usage {
            input_tokens,
            output_tokens,
            model: cfg.model.profile.clone().unwrap_or_default(),
        }),
        other => Some(other),
    }
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, fields(agent_id = %cfg.agent_id))]
pub async fn run_agent<T: ToolTransport>(
    llm: &LlmRegistry,
    transport: &T,
    cfg: &AgentConfig,
    history: &[ConvTurn],
    user_message: &str,
    context: Option<&serde_json::Value>,
    tx: &mpsc::Sender<AgentEvent>,
    retry: RetryConfig,
    cancel: &CancellationToken,
) -> Result<AgentRun> {
    if tx
        .send(AgentEvent::Started {
            agent_id: cfg.agent_id.clone(),
        })
        .await
        .is_err()
    {
        return Ok(empty_run());
    }

    let backend = llm.backend_for(&cfg.model)?;
    // Resolve declared tools against the live manifest.
    let manifest: Manifest = if cfg.tools.is_empty() {
        Manifest { tools: vec![] }
    } else {
        transport.list_tools().await?
    };
    let base = base_request(cfg, build_tool_defs(cfg, &manifest)?);

    let mut messages: Vec<Message> = assemble_messages(history, user_message);
    let mut conv_turns: Vec<ConvTurn> = vec![ConvTurn::text(ConvRole::User, user_message)];
    let mut usage = UsageTotals::default();

    for _ in 0..cfg.guardrails.max_iterations.unwrap_or(4) {
        let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(64);
        let req = ChatRequest {
            messages: messages.clone(),
            ..base.clone()
        };
        let inference = stream_chat(backend.as_ref(), req, &llm_tx, retry);
        tokio::pin!(inference);

        let mut turn = Turn::default();
        loop {
            tokio::select! {
                // Checked first: a cancel must win over a reply that is also ready.
                biased;
                _ = cancel.cancelled() => {
                    return Ok(aborted_run(conv_turns, usage, "cancelled"));
                }
                result = &mut inference => {
                    // Return the turns, not the error: a remembering stage must
                    // still record the question (see `aborted_run`).
                    if let Err(e) = result {
                        return Ok(aborted_run(conv_turns, usage, &e.to_string()));
                    }
                    // Drain any remaining events after the future resolved.
                    while let Ok(event) = llm_rx.try_recv() {
                        if let Some(ae) = fold_event(&mut turn, &mut usage, cfg, event) {
                            let _ = tx.send(ae).await;
                        }
                    }
                    break;
                }
                Some(event) = llm_rx.recv() => {
                    if let Some(ae) = fold_event(&mut turn, &mut usage, cfg, event) {
                        if tx.send(ae).await.is_err() {
                            return Ok(aborted_run(conv_turns, usage, "client disconnected"));
                        }
                    }
                }
            }
        }

        if turn.tool_calls.is_empty() {
            let output = match finalize_output(turn.text, cfg.guardrails.output_schema.as_ref()) {
                Ok(output) => output,
                Err(e) => return Ok(aborted_run(conv_turns, usage, &e.to_string())),
            };
            conv_turns.push(ConvTurn::text(ConvRole::Assistant, output.message.clone()));
            let _ = tx
                .send(AgentEvent::ResponseComplete {
                    output: output.clone(), // clone: event owns its copy; caller gets the return
                })
                .await;
            return Ok(AgentRun {
                output,
                turns: conv_turns,
                aborted: None,
                usage,
            });
        }

        // Append the assistant turn (opaque blocks + text + tool calls) to the
        // request messages, and its text + tool calls to the conversation
        // turns before the tools run, so a run cancelled mid-call still
        // records what the model asked for. Opaque blocks stay out of the
        // conversation turns: they are only valid within this tool loop.
        messages.push(assistant_message(&turn));
        conv_turns.push(ConvTurn {
            role: ConvRole::Assistant,
            content: turn.text.clone(),
            tool_calls: turn
                .tool_calls
                .iter()
                .map(|c| flux_core::conversation::ToolCallRecord {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    args: c.args.clone(),
                })
                .collect(),
            tool_results: Vec::new(),
        });

        // Execute the turn's tool calls concurrently; tool errors are fed back
        // to the model as results rather than aborting the run.
        let tool_calls = run_tool_calls(
            transport,
            &manifest,
            &cfg.tools,
            context,
            &turn.tool_calls,
            tx,
        );
        // A tool call already received by its server may still complete
        // there; cancelling only stops Flux waiting for it.
        let results = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(aborted_run(conv_turns, usage, "cancelled")),
            results = tool_calls => results,
        };
        for (call, result) in turn.tool_calls.iter().zip(&results) {
            messages.push(tool_result_message(call, result));
        }

        // The results fed back, as one user turn.
        conv_turns.push(ConvTurn {
            role: ConvRole::User,
            content: String::new(),
            tool_calls: Vec::new(),
            tool_results: turn
                .tool_calls
                .iter()
                .zip(&results)
                .map(|(c, r)| flux_core::conversation::ToolResultRecord {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    result: r.clone(),
                })
                .collect(),
        });
    }

    tracing::warn!(agent_id = %cfg.agent_id, "max_iterations exceeded");
    let _ = tx
        .send(AgentEvent::Failed {
            error: "max_iterations exceeded".into(),
        })
        .await;
    Ok(aborted_run(conv_turns, usage, "max_iterations exceeded"))
}

// ---------------------------------------------------------------------------
// Output finalisation
// ---------------------------------------------------------------------------

/// Turn the agent's final text into an `AgentOutput`. With a schema, the text
/// must parse as JSON and validate against it (fail-fast); without one, the
/// text is returned as-is with no structured value.
pub fn finalize_output(
    text: String,
    output_schema: Option<&serde_json::Value>,
) -> Result<AgentOutput> {
    match output_schema {
        None => Ok(AgentOutput {
            message: text,
            structured: None,
        }),
        Some(schema) => {
            let value: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| FluxError::Decode(e.to_string()))?;
            // Same builder as tool-arg validation: external `$ref`s are refused
            // rather than fetched. An `output_schema` comes from first-party
            // agent YAML, so this is consistency rather than a live exposure.
            //
            // Cached, because this runs on every response of every agent that
            // declares a schema — recompiling per response made the check cost
            // scale with traffic for a document that never changes.
            let validator = flux_core::schema::cached_validator(schema)
                .map_err(|e| FluxError::Config(format!("invalid output_schema: {e}")))?;
            if let Err(error) = validator.validate(&value) {
                return Err(FluxError::Config(format!(
                    "agent output failed schema validation: {error}"
                )));
            }
            Ok(AgentOutput {
                message: text,
                structured: Some(value),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// NoopTransport — used when no tools are declared (e.g. CLI smoke-test)
// ---------------------------------------------------------------------------

/// No-op transport used by the CLI when an agent declares no tools.
///
/// The runner never invokes `list_tools` or `call_tool` when `cfg.tools` is
/// empty, so this transport is never actually called — it satisfies the generic
/// bound without establishing any network connection.
pub struct NoopTransport;

impl ToolTransport for NoopTransport {
    async fn list_tools(&self) -> Result<Manifest> {
        Err(FluxError::Config(
            "NoopTransport: no tool endpoint configured".into(),
        ))
    }

    async fn call_tool(&self, _call: &TcCall) -> Result<flux_tool_client::ToolResult> {
        Err(FluxError::Config(
            "NoopTransport: no tool endpoint configured".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flux_config::{AgentConfig, Guardrails, ModelConfig};
    use flux_llm::{
        Content, LlmEvent, LlmRegistry, MockBackend, MockTurn, OutputFormat, ProviderBlock, Role,
        Usage,
    };
    use serde_json::json;

    use flux_tool_client::ToolResult as TcResult;
    use std::sync::{Arc, Mutex};

    /// Fake transport: returns a canned ok result per tool name, or an error for
    /// names in `fail`. Records the names it was actually asked to call.
    struct FakeTransport {
        fail: Vec<String>,
        called: Mutex<Vec<String>>,
    }

    impl ToolTransport for FakeTransport {
        async fn list_tools(&self) -> Result<Manifest> {
            Ok(weather_manifest())
        }
        async fn call_tool(&self, call: &TcCall) -> Result<TcResult> {
            self.called.lock().unwrap().push(call.name.clone());
            if self.fail.contains(&call.name) {
                Err(FluxError::Provider(format!("boom: {}", call.name)))
            } else {
                Ok(TcResult::new(json!({ "ok": call.name })))
            }
        }
    }

    fn captured(name: &str, args: serde_json::Value) -> CapturedToolCall {
        CapturedToolCall {
            id: format!("id-{name}"),
            name: name.into(),
            args,
            invalid_args: None,
        }
    }

    #[tokio::test]
    async fn tool_calls_run_and_preserve_order_with_error_feedback() {
        let transport = FakeTransport {
            fail: vec!["b".into()],
            called: Mutex::new(vec![]),
        };
        let manifest = Manifest { tools: vec![] }; // no specs → no validation
        let allowed = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let calls = vec![
            captured("a", json!({})),
            captured("b", json!({})),
            captured("c", json!({})),
        ];
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);
        let results = run_tool_calls(&transport, &manifest, &allowed, None, &calls, &tx).await;
        drop(tx);

        // one result per call, in order
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], json!({ "ok": "a" }));
        assert_eq!(results[1]["error"], "provider error: boom: b");
        assert_eq!(results[2], json!({ "ok": "c" }));

        // events: 3 ToolCall then 3 ToolResult
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        let calls_seen = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolCall { .. }))
            .count();
        let results_seen = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .count();
        assert_eq!(calls_seen, 3);
        assert_eq!(results_seen, 3);
    }

    #[tokio::test]
    async fn a_call_to_an_undeclared_tool_is_refused_without_calling_transport() {
        let transport = FakeTransport {
            fail: vec![],
            called: Mutex::new(vec![]),
        };
        // The transport serves every backend's tools; the agent declared only
        // get_weather. A model that names another backend's tool must not reach it.
        let manifest = weather_manifest();
        let allowed = vec!["get_weather".to_string()];
        let calls = vec![captured("delete_rows", json!({}))];
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);
        let results = run_tool_calls(&transport, &manifest, &allowed, None, &calls, &tx).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["error"], "tool 'delete_rows' is not available");
        assert!(transport.called.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn invalid_args_feed_back_without_calling_transport() {
        let transport = FakeTransport {
            fail: vec![],
            called: Mutex::new(vec![]),
        };
        // get_weather requires `location`; omit it → validation fails.
        let manifest = weather_manifest();
        let allowed = vec!["get_weather".to_string()];
        let calls = vec![captured("get_weather", json!({}))];
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);
        let results = run_tool_calls(&transport, &manifest, &allowed, None, &calls, &tx).await;

        assert_eq!(results.len(), 1);
        assert!(
            results[0].get("error").is_some(),
            "expected validation error fed back, got {:?}",
            results[0]
        );
        // transport.call_tool must NOT have been invoked for the invalid call
        assert!(transport.called.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_successful_result_with_an_error_key_is_passed_through() {
        struct ErrorKeyedTransport;
        impl ToolTransport for ErrorKeyedTransport {
            async fn list_tools(&self) -> Result<Manifest> {
                Ok(Manifest { tools: vec![] })
            }
            async fn call_tool(&self, _call: &TcCall) -> Result<TcResult> {
                Ok(TcResult::new(json!({ "error": null, "data": 1 })))
            }
        }
        let calls = vec![captured("probe", json!({}))];
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);
        let results = run_tool_calls(
            &ErrorKeyedTransport,
            &Manifest { tools: vec![] },
            &["probe".to_string()],
            None,
            &calls,
            &tx,
        )
        .await;
        assert_eq!(results[0], json!({ "error": null, "data": 1 }));
    }

    #[tokio::test]
    async fn a_cancelled_agent_returns_its_turns() {
        let mut cfg = make_cfg(vec![]);
        cfg.model.provider = "mock".into();
        cfg.model.mock_response = Some("done".into());
        let llm = LlmRegistry::new();
        let transport = FakeTransport {
            fail: vec![],
            called: Mutex::new(vec![]),
        };
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let run = run_agent(
            &llm,
            &transport,
            &cfg,
            &[],
            "question",
            None,
            &tx,
            RetryConfig::default(),
            &cancel,
        )
        .await
        .unwrap();
        assert_eq!(run.aborted.as_deref(), Some("cancelled"));
        assert_eq!(run.turns.len(), 2);
        assert_eq!(run.turns[0].content, "question");
        assert_eq!(run.turns[1].content, "[run stopped: cancelled]");
    }

    /// A registry whose `scripted` provider plays `turns`; the Arc is kept so
    /// the test can read the requests it received.
    fn scripted(turns: Vec<MockTurn>) -> (LlmRegistry, Arc<MockBackend>) {
        let backend = Arc::new(MockBackend::scripted(turns));
        let mut llm = LlmRegistry::new();
        llm.register("scripted", backend.clone());
        (llm, backend)
    }

    fn scripted_cfg(tools: Vec<String>) -> AgentConfig {
        let mut cfg = make_cfg(tools);
        cfg.model.provider = "scripted".into();
        cfg
    }

    fn fake_transport() -> FakeTransport {
        FakeTransport {
            fail: vec![],
            called: Mutex::new(vec![]),
        }
    }

    async fn run_with(
        llm: &LlmRegistry,
        transport: &FakeTransport,
        cfg: &AgentConfig,
    ) -> (AgentRun, Vec<AgentEvent>) {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentEvent>(256);
        let run = run_agent(
            llm,
            transport,
            cfg,
            &[],
            "question",
            None,
            &tx,
            RetryConfig::default(),
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        drop(tx);
        let mut events = vec![];
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        (run, events)
    }

    fn weather_call(id: &str) -> LlmEvent {
        LlmEvent::ToolCall {
            id: id.into(),
            name: "get_weather".into(),
            args: json!({ "location": "SF" }),
        }
    }

    fn usage(input: u32, output: u32) -> LlmEvent {
        LlmEvent::Usage(Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            model: None,
        })
    }

    #[tokio::test]
    async fn usage_is_reported_once_per_llm_call_and_summed() {
        let (llm, _backend) = scripted(vec![
            MockTurn::Events(vec![weather_call("c1"), usage(10, 5)]),
            MockTurn::Events(vec![LlmEvent::Text("sunny".into()), usage(3, 2)]),
        ]);
        let cfg = scripted_cfg(vec!["get_weather".into()]);
        let (run, events) = run_with(&llm, &fake_transport(), &cfg).await;

        let usages: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::Usage { .. }))
            .collect();
        assert_eq!(
            usages,
            vec![
                &AgentEvent::Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    model: "claude-opus-4-8".into(),
                },
                &AgentEvent::Usage {
                    input_tokens: Some(3),
                    output_tokens: Some(2),
                    model: "claude-opus-4-8".into(),
                },
            ]
        );
        assert_eq!(
            run.usage,
            UsageTotals {
                input_tokens: Some(13),
                output_tokens: Some(7),
            }
        );
    }

    #[test]
    fn usage_totals_leave_a_missing_side_unset() {
        let mut totals = UsageTotals::default();
        totals.add(&Usage {
            input_tokens: Some(4),
            output_tokens: None,
            model: None,
        });
        totals.add(&Usage::default());
        assert_eq!(totals.input_tokens, Some(4));
        assert_eq!(totals.output_tokens, None);
    }

    #[tokio::test]
    async fn thinking_blocks_are_replayed_in_the_tool_loop_but_not_remembered() {
        let thinking = ProviderBlock {
            provider: "anthropic".into(),
            data: json!({ "type": "thinking", "thinking": "hmm", "signature": "sig-7f3a" }),
        };
        let (llm, backend) = scripted(vec![
            MockTurn::Events(vec![
                LlmEvent::Thought("hmm".into()),
                LlmEvent::Opaque(thinking.clone()),
                weather_call("c1"),
            ]),
            MockTurn::Text("sunny".into()),
        ]);
        let cfg = scripted_cfg(vec!["get_weather".into()]);
        let (run, _events) = run_with(&llm, &fake_transport(), &cfg).await;
        assert!(run.aborted.is_none(), "{:?}", run.aborted);

        let requests = backend.requests();
        assert_eq!(requests.len(), 2);
        let assistant = requests[1]
            .messages
            .iter()
            .find(|m| m.role == Role::Assistant)
            .expect("the tool-calling assistant turn is replayed");
        assert_eq!(assistant.content[0], Content::Opaque(thinking));
        assert!(
            !format!("{:?}", run.turns).contains("sig-7f3a"),
            "thinking blocks must not reach conversation memory: {:?}",
            run.turns
        );
    }

    #[tokio::test]
    async fn malformed_tool_arguments_are_fed_back_as_an_error() {
        let (llm, backend) = scripted(vec![
            MockTurn::Events(vec![LlmEvent::InvalidToolCall {
                id: "c1".into(),
                name: "get_weather".into(),
                error: "expected value".into(),
            }]),
            MockTurn::Text("sorry".into()),
        ]);
        let cfg = scripted_cfg(vec!["get_weather".into()]);
        let transport = fake_transport();
        let (run, _events) = run_with(&llm, &transport, &cfg).await;
        assert!(run.aborted.is_none(), "{:?}", run.aborted);
        assert!(transport.called.lock().unwrap().is_empty());

        let requests = backend.requests();
        let expected = Content::ToolResult {
            id: "c1".into(),
            name: "get_weather".into(),
            result: json!({ "error": "invalid tool arguments: expected value" }),
        };
        assert!(
            requests[1]
                .messages
                .iter()
                .any(|m| m.content.contains(&expected)),
            "{:?}",
            requests[1].messages
        );
    }

    #[tokio::test]
    async fn validate_mode_sends_no_schema() {
        let (llm, backend) = scripted(vec![MockTurn::Text("{\"a\":1}".into())]);
        let mut cfg = scripted_cfg(vec![]);
        cfg.guardrails.output_schema = Some(json!({ "type": "object" }));
        cfg.guardrails.output_schema_mode = flux_config::OutputSchemaMode::Validate;
        let (run, _events) = run_with(&llm, &fake_transport(), &cfg).await;
        assert!(run.aborted.is_none(), "{:?}", run.aborted);
        assert_eq!(backend.requests()[0].output, OutputFormat::Text);
    }

    #[test]
    fn the_request_carries_the_configured_model_settings() {
        let mut cfg = make_cfg(vec![]);
        cfg.model.max_tokens = Some(777);
        cfg.model.temperature = Some(0.25);
        let request = base_request(&cfg, vec![]);
        assert_eq!(request.model, "claude-opus-4-8");
        assert_eq!(request.system, build_system_prompt(&cfg));
        assert_eq!(request.max_tokens, Some(777));
        assert_eq!(request.temperature, Some(0.25));
        assert!(request.messages.is_empty());
    }

    #[test]
    fn reasoning_is_none_unless_effort_or_budget_is_set() {
        let mut cfg = make_cfg(vec![]);
        assert_eq!(base_request(&cfg, vec![]).reasoning, None);

        cfg.model.reasoning_effort = Some("high".into());
        assert_eq!(
            base_request(&cfg, vec![]).reasoning,
            Some(Reasoning {
                effort: Some("high".into()),
                budget_tokens: None,
            })
        );

        cfg.model.reasoning_effort = None;
        cfg.model.thinking_budget_tokens = Some(2048);
        assert_eq!(
            base_request(&cfg, vec![]).reasoning,
            Some(Reasoning {
                effort: None,
                budget_tokens: Some(2048),
            })
        );
    }

    #[test]
    fn a_constrained_output_schema_becomes_json_output() {
        let mut cfg = make_cfg(vec![]);
        assert_eq!(base_request(&cfg, vec![]).output, OutputFormat::Text);

        let schema = json!({ "type": "object" });
        cfg.guardrails.output_schema = Some(schema.clone());
        assert_eq!(
            base_request(&cfg, vec![]).output,
            OutputFormat::Json { schema }
        );
    }

    #[test]
    fn tool_defs_come_from_the_manifest() {
        let cfg = make_cfg(vec!["get_weather".into()]);
        let tools = build_tool_defs(&cfg, &weather_manifest()).unwrap();
        let request = base_request(&cfg, tools);
        assert_eq!(
            request.tools,
            vec![ToolDef {
                name: "get_weather".into(),
                description: "Get weather for a location".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"]
                }),
            }]
        );
    }

    #[tokio::test]
    async fn invalid_args_are_rejected_before_the_allowed_list_check() {
        let transport = fake_transport();
        // `delete_rows` is not in the allowed list, but malformed arguments
        // are reported first.
        let mut call = captured("delete_rows", json!({}));
        call.invalid_args = Some("expected value".into());
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(64);
        let results = run_tool_calls(
            &transport,
            &weather_manifest(),
            &["get_weather".to_string()],
            None,
            &[call],
            &tx,
        )
        .await;
        assert_eq!(
            results[0]["error"],
            "invalid tool arguments: expected value"
        );
        assert!(transport.called.lock().unwrap().is_empty());
    }

    fn make_cfg(tools: Vec<String>) -> AgentConfig {
        AgentConfig {
            agent_id: "test-agent".into(),
            display_name: "Test Agent".into(),
            model: ModelConfig {
                provider: "anthropic".into(),
                profile: Some("claude-opus-4-8".into()),
                max_tokens: Some(1024),
                reasoning_effort: None,
                thinking_budget_tokens: None,
                temperature: None,
                mock_response: None,
            },
            persona: None,
            system_prompt: "You are helpful.".into(),
            context: vec![],
            tools,
            guardrails: Guardrails {
                max_iterations: Some(4),
                output_schema: None,
                ..Default::default()
            },
        }
    }

    fn weather_manifest() -> Manifest {
        use flux_tool_client::ToolSpec;
        Manifest {
            tools: vec![ToolSpec {
                name: "get_weather".into(),
                description: "Get weather for a location".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"]
                }),
            }],
        }
    }

    #[test]
    fn turn_accumulates_text_and_tool_calls() {
        let mut turn = Turn::default();

        // Thought → forwarded as AgentEvent
        let ae = turn.on_event(LlmEvent::Thought("planning".into()));
        assert_eq!(
            ae,
            Some(AgentEvent::Thought {
                text: "planning".into()
            })
        );

        // Text chunks are emitted as deltas AND accumulate into turn.text
        assert_eq!(
            turn.on_event(LlmEvent::Text("Hello ".into())),
            Some(AgentEvent::Text {
                delta: "Hello ".into()
            })
        );
        assert_eq!(
            turn.on_event(LlmEvent::Text("world".into())),
            Some(AgentEvent::Text {
                delta: "world".into()
            })
        );

        // ToolCall buffered (no immediate event)
        assert_eq!(
            turn.on_event(LlmEvent::ToolCall {
                id: "t1".into(),
                name: "get_weather".into(),
                args: json!({ "location": "SF" }),
            }),
            None
        );

        // Done produces no event
        assert_eq!(turn.on_event(LlmEvent::Done), None);

        assert_eq!(turn.text, "Hello world");
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "get_weather");
        assert_eq!(turn.tool_calls[0].id, "t1");
        assert_eq!(turn.tool_calls[0].args, json!({ "location": "SF" }));
    }

    #[test]
    fn history_message_rebuilds_tool_call_blocks() {
        let turn = ConvTurn {
            role: ConvRole::Assistant,
            content: "thinking".into(),
            tool_calls: vec![flux_core::conversation::ToolCallRecord {
                id: "c1".into(),
                name: "echo".into(),
                args: serde_json::json!({ "text": "hi" }),
            }],
            tool_results: Vec::new(),
        };
        let msg = history_message(&turn).expect("turn has content");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 2); // text block + tool_call block
    }

    #[test]
    fn history_message_rebuilds_tool_result_blocks() {
        let turn = ConvTurn {
            role: ConvRole::User,
            content: String::new(),
            tool_calls: Vec::new(),
            tool_results: vec![flux_core::conversation::ToolResultRecord {
                id: "c1".into(),
                name: "echo".into(),
                result: serde_json::json!({ "echoed": "hi" }),
            }],
        };
        let msg = history_message(&turn).expect("turn has a tool result");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 1); // no text, one tool_result block
    }

    #[test]
    fn history_message_of_plain_text_turn_is_one_text_block() {
        let turn = ConvTurn::text(ConvRole::User, "hello");
        let msg = history_message(&turn).expect("turn has content");
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn history_message_skips_a_turn_with_no_content_blocks() {
        // A model reply that came back empty, or a legacy row predating the
        // tool columns. Emitting it would mean a message with an empty
        // `content` array, which providers reject — poisoning every later turn
        // of the conversation rather than losing this one line.
        let turn = ConvTurn::text(ConvRole::Assistant, "");
        assert!(history_message(&turn).is_none());
    }

    #[test]
    fn assemble_messages_drops_empty_history_turns() {
        let history = vec![
            ConvTurn::text(ConvRole::User, "a"),
            ConvTurn::text(ConvRole::Assistant, ""), // would be an empty message
            ConvTurn::text(ConvRole::User, "b"),
        ];
        let msgs = assemble_messages(&history, "now");
        assert_eq!(msgs.len(), 3); // two surviving history turns + current
        assert!(msgs.iter().all(|m| !m.content.is_empty()));
    }

    #[test]
    fn aborted_run_keeps_its_turns_and_closes_on_an_assistant_turn() {
        // The user's question must survive a run that gave up, or the next turn
        // shows no trace it was ever asked.
        let turns = vec![
            ConvTurn::text(ConvRole::User, "q"),
            ConvTurn {
                role: ConvRole::Assistant,
                content: String::new(),
                tool_calls: vec![flux_core::conversation::ToolCallRecord {
                    id: "c1".into(),
                    name: "t".into(),
                    args: serde_json::json!({}),
                }],
                tool_results: Vec::new(),
            },
        ];
        let run = aborted_run(turns, UsageTotals::default(), "max_iterations exceeded");
        assert!(run.output.message.is_empty());
        assert_eq!(run.turns.len(), 3);
        assert_eq!(run.turns[0].content, "q");
        // Closing turn is an assistant turn with real text, so the log cannot
        // end on a user/tool-result turn and cannot yield an empty message.
        let last = run.turns.last().expect("closing turn");
        assert_eq!(last.role, ConvRole::Assistant);
        assert!(!last.content.is_empty());
        assert!(history_message(last).is_some());
        assert_eq!(run.aborted.as_deref(), Some("max_iterations exceeded"));
    }

    #[test]
    fn assemble_messages_orders_history_then_current() {
        // empty history → just the current user message
        let only = assemble_messages(&[], "hello");
        assert_eq!(only.len(), 1);
        assert!(matches!(only[0].role, Role::User));

        // history (oldest-first) then current message last
        let history = vec![
            ConvTurn::text(ConvRole::User, "a"),
            ConvTurn::text(ConvRole::Assistant, "b"),
        ];
        let msgs = assemble_messages(&history, "c");
        assert_eq!(msgs.len(), 3);
        assert!(matches!(msgs[0].role, Role::User));
        assert!(matches!(msgs[1].role, Role::Assistant));
        assert!(matches!(msgs[2].role, Role::User));

        let texts: Vec<String> = msgs
            .iter()
            .map(|m| match &m.content[0] {
                Content::Text(t) => t.clone(),
                _ => panic!("expected text content"),
            })
            .collect();
        assert_eq!(texts, vec!["a", "b", "c"]);
    }

    #[test]
    fn build_tool_defs_resolves_known_tool() {
        let cfg = make_cfg(vec!["get_weather".into()]);
        let manifest = weather_manifest();
        let tools = build_tool_defs(&cfg, &manifest).expect("should succeed");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].parameters, manifest.tools[0].parameters);
    }

    #[test]
    fn build_tool_defs_rejects_missing_tool() {
        let cfg = make_cfg(vec!["missing_tool".into()]);
        let manifest = weather_manifest();
        match build_tool_defs(&cfg, &manifest) {
            Err(FluxError::Config(msg)) => {
                assert!(
                    msg.contains("missing_tool"),
                    "error should name the tool: {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn build_tool_defs_empty_returns_empty() {
        let cfg = make_cfg(vec![]);
        let tools = build_tool_defs(&cfg, &Manifest { tools: vec![] }).expect("should succeed");
        assert!(tools.is_empty());
    }

    #[test]
    fn finalize_no_schema_returns_text() {
        let out = finalize_output("hello world".into(), None).unwrap();
        assert_eq!(out.message, "hello world");
        assert!(out.structured.is_none());
    }

    #[test]
    fn finalize_valid_json_against_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "sentiment": { "type": "string" } },
            "required": ["sentiment"]
        });
        let out = finalize_output(r#"{"sentiment":"positive"}"#.into(), Some(&schema)).unwrap();
        assert_eq!(out.structured.unwrap()["sentiment"], "positive");
    }

    #[test]
    fn finalize_rejects_non_conforming_json() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "sentiment": { "type": "string" } },
            "required": ["sentiment"]
        });
        let err = finalize_output(r#"{"other":1}"#.into(), Some(&schema));
        assert!(err.is_err());
    }

    #[test]
    fn finalize_rejects_unparseable_json() {
        let schema = serde_json::json!({ "type": "object" });
        let err = finalize_output("not json".into(), Some(&schema));
        assert!(err.is_err());
    }
}
