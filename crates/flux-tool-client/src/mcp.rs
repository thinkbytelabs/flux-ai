//! MCP tool transport over stdio.
//!
//! This is the ONLY module permitted to name an `rmcp` type. Everything it
//! exposes is in Flux's own vocabulary (`ToolSpec`, `ToolCall`, `ToolResult`),
//! so an rmcp major-version bump is contained to this file.

use crate::{Manifest, ToolSpec, ToolTransport};
use flux_core::error::{FluxError, Result};
use std::time::Duration;

/// Budget for spawn plus the MCP `initialize` handshake. A server that spawns
/// but never answers would otherwise hang boot forever with no diagnostic:
/// gRPC has `GRPC_CONNECT_TIMEOUT`, stdio has nothing equivalent.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// Budget for the boot-time `tools/list`, including every page of it.
///
/// `HANDSHAKE_TIMEOUT` bounds `initialize` only. A server that answers the
/// handshake and then goes silent used to hang the very next request instead —
/// same unbounded wait, one RPC later, and still during boot, so
/// `flux-server`/`run_workflow` never finished starting and printed nothing.
const LIST_TOOLS_TIMEOUT: Duration = Duration::from_secs(30);

/// Budget for one `tools/call`.
///
/// Deliberately generous, and larger than the 60s several MCP clients default
/// to: an MCP tool does real work, and this is a backstop against a wedged
/// server rather than a latency target. It is a constant because nothing has
/// asked for it to vary — a tool that legitimately runs longer is the signal to
/// make it configurable.
const CALL_TOOL_TIMEOUT: Duration = Duration::from_secs(120);

/// Variables every MCP child gets from Flux's environment when set: enough to
/// find and run `npx`/`uvx`, pick a locale, find writable cache dirs and trust
/// a corporate CA, and nothing that holds a secret. Proxy variables are left
/// out because their URLs can carry credentials; list them in
/// [`McpServer::env`] when a server needs them.
pub const BASE_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
    "REQUESTS_CA_BUNDLE",
];

/// How to launch one MCP server.
#[derive(Debug, Clone, PartialEq)]
pub struct McpServer {
    pub command: String,
    pub args: Vec<String>,
    /// Names of Flux's environment variables to pass to the child, on top of
    /// [`BASE_ENV`]. Nothing else is inherited.
    pub env: Vec<String>,
}

impl McpServer {
    /// Human-readable identity for error messages. With several servers
    /// configured, a bare "connection closed" names no culprit.
    fn label(&self) -> String {
        if self.args.is_empty() {
            self.command.clone()
        } else {
            format!("{} {}", self.command, self.args.join(" "))
        }
    }
}

#[derive(Debug)]
pub struct McpTransport {
    service: rmcp::service::RunningService<rmcp::RoleClient, ()>,
    label: String,
    child_pid: Option<u32>,
}

impl McpTransport {
    /// Spawn the server and complete the MCP handshake.
    pub async fn spawn(server: &McpServer) -> Result<Self> {
        use rmcp::transport::child_process::TokioChildProcess;
        use rmcp::ServiceExt;
        use std::process::Stdio;

        let label = server.label();

        let mut cmd = tokio::process::Command::new(&server.command);
        cmd.args(&server.args);
        // An MCP server is third-party code: it must not see Flux's LLM keys or
        // database URLs unless they are named for it.
        cmd.env_clear();
        for var in BASE_ENV
            .iter()
            .copied()
            .chain(server.env.iter().map(String::as_str))
        {
            if let Some(value) = std::env::var_os(var) {
                cmd.env(var, value);
            }
        }

        // stderr is inherited on purpose. If it were piped and never drained,
        // a chatty server would fill the pipe and block forever — a hang that
        // looks like a slow tool. The builder already defaults to inherit;
        // stating it here stops that default from drifting silently.
        let (transport, _stderr) = TokioChildProcess::builder(cmd)
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                FluxError::Provider(format!(
                    "spawning MCP server `{label}` failed: {e} \
                     (is `{}` on PATH?)",
                    server.command
                ))
            })?;

        let child_pid = transport.id();

        let service = tokio::time::timeout(HANDSHAKE_TIMEOUT, ().serve(transport))
            .await
            .map_err(|_| {
                FluxError::Provider(format!(
                    "MCP server `{label}` did not complete the initialize handshake within {}s",
                    HANDSHAKE_TIMEOUT.as_secs()
                ))
            })?
            .map_err(|e| {
                FluxError::Provider(format!("MCP handshake with `{label}` failed: {e}"))
            })?;

        Ok(McpTransport {
            service,
            label,
            child_pid,
        })
    }

    /// PID of the child, for lifecycle assertions in tests.
    pub fn child_pid(&self) -> Option<u32> {
        self.child_pid
    }
}

/// Bound an MCP request in time.
///
/// stdio offers nothing equivalent to a gRPC connect timeout. A child that is
/// alive but never answers produces no error, no close and no EOF — the await
/// simply never returns, so without this every request is an unbounded wait.
///
/// The timeout abandons the response rather than sending MCP's
/// `notifications/cancelled`. rmcp emits that only from its own per-request
/// timeout, which is reachable only by hand-building requests through
/// `send_cancellable_request`: that would forfeit `list_all_tools`' pagination
/// and pull four more rmcp types into this deliberately sealed module. So a
/// server may keep working on a request nobody is waiting for — accepted,
/// because for the failure this guards (a server that never answers at all)
/// cancelling changes nothing. Request ids are never reused, so a late reply is
/// discarded and the transport stays usable for the next call.
async fn within<T>(
    budget: Duration,
    request: &str,
    label: &str,
    call: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::time::timeout(budget, call).await.map_err(|_| {
        FluxError::Provider(format!(
            "MCP server `{label}` did not answer `{request}` within {}s",
            budget.as_secs()
        ))
    })?
}

impl ToolTransport for McpTransport {
    async fn list_tools(&self) -> Result<Manifest> {
        // `list_all_tools` paginates internally, so a server with many tools
        // is not silently truncated. The budget covers every page, so a server
        // that pages forever is bounded too.
        let tools = within(LIST_TOOLS_TIMEOUT, "tools/list", &self.label, async {
            self.service.list_all_tools().await.map_err(|e| {
                FluxError::Provider(format!(
                    "listing tools from MCP server `{}`: {e}",
                    self.label
                ))
            })
        })
        .await?;

        let tools = tools
            .into_iter()
            .map(|t| ToolSpec {
                name: t.name.to_string(),
                // gRPC yields "" for a missing description; match that.
                description: t.description.map(|d| d.to_string()).unwrap_or_default(),
                parameters: serde_json::Value::Object((*t.input_schema).clone()),
            })
            .collect();

        Ok(Manifest { tools })
    }

    #[tracing::instrument(skip_all, fields(tool = %call.name, server = %self.label))]
    async fn call_tool(&self, call: &crate::ToolCall) -> Result<crate::ToolResult> {
        // MCP's `tools/call` has no field for Flux's out-of-band context. Erroring
        // would make any mixed deployment unusable — a run that sets context for
        // its gRPC tools would die on its first MCP tool — and dropping it in
        // silence is the real trap. So: drop it, loudly.
        if call.context.is_some() {
            tracing::warn!(
                tool = %call.name,
                server = %self.label,
                "context is a gRPC-only field; MCP has no equivalent, so it was dropped"
            );
        }

        let mut params = rmcp::model::CallToolRequestParams::new(call.name.clone());
        if let Some(arguments) = arguments_for(&call.name, &call.args)? {
            params = params.with_arguments(arguments);
        }

        let result = within(CALL_TOOL_TIMEOUT, "tools/call", &self.label, async {
            self.service.call_tool(params).await.map_err(|e| {
                FluxError::Provider(format!(
                    "calling MCP tool `{}` on `{}`: {e}",
                    call.name, self.label
                ))
            })
        })
        .await?;

        map_call_result(&call.name, result)
    }
}

/// Convert Flux's arbitrary-JSON args into MCP's `arguments` field.
///
/// MCP requires arguments to be a JSON object; the gRPC transport never cared
/// because it shipped a string. Schema validation normally rejects a non-object
/// first (an MCP tool's `input_schema` is always `type: object`), so this is the
/// backstop for when it does not — and it must not panic, because this crate
/// denies `unwrap_used`.
fn arguments_for(tool: &str, args: &serde_json::Value) -> Result<Option<rmcp::model::JsonObject>> {
    match args {
        serde_json::Value::Object(map) => Ok(Some(map.clone())),
        serde_json::Value::Null => Ok(None),
        other => {
            let kind = match other {
                serde_json::Value::Array(_) => "an array",
                serde_json::Value::String(_) => "a string",
                serde_json::Value::Number(_) => "a number",
                serde_json::Value::Bool(_) => "a boolean",
                // Object and Null are handled above.
                _ => "a non-object value",
            };
            Err(FluxError::Config(format!(
                "MCP tool `{tool}` requires object arguments, got {kind}"
            )))
        }
    }
}

/// Map an MCP `CallToolResult` onto Flux's `ToolResult`.
///
/// The shape is FIXED — `content`, `text`, `structured_content` are always
/// present — so a workflow's expression does not break when a server starts
/// returning structured output. Text is never parsed as JSON: a tool returning
/// the literal `123` must stay a string.
fn map_call_result(tool: &str, result: rmcp::model::CallToolResult) -> Result<crate::ToolResult> {
    let text = result
        .content
        .iter()
        .filter_map(|block| block.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    // A tool that ran and failed is a runtime failure, the same class as a gRPC
    // tool returning an error status — so `Provider`, matching the error
    // `arguments_for` returns as `Config` for a pre-dispatch defect. The variant
    // only records the *class* of defect for diagnostics: fatality is decided by
    // which `StageOutcome` arm the orchestrator picks, not by the variant. Any
    // error out of `call_tool` (`Provider` here, `Config` from `arguments_for`)
    // becomes `StageOutcome::Failed`, which `on_error: continue` can absorb;
    // only the pre-dispatch checks that run before `call_tool` produce
    // `StageOutcome::Invalid`, which is always fatal.
    if result.is_error.unwrap_or(false) {
        let detail = if text.is_empty() {
            "<no message>".to_string()
        } else {
            text
        };
        return Err(FluxError::Provider(format!(
            "MCP tool `{tool}` reported an error: {detail}"
        )));
    }

    let content = serde_json::to_value(&result.content)
        .map_err(|e| FluxError::Decode(format!("encoding MCP content for `{tool}`: {e}")))?;

    // Kinds of the blocks that contribute nothing to `text`, named so the
    // model-facing projection can mention them without carrying their payloads
    // (an image's base64 data would dwarf everything else in the prompt).
    let non_text_kinds: Vec<&str> = result
        .content
        .iter()
        .filter(|block| block.as_text().is_none())
        .map(|block| {
            if block.as_image().is_some() {
                "image"
            } else if block.as_audio().is_some() {
                "audio"
            } else if block.as_resource().is_some() {
                "resource"
            } else if block.as_resource_link().is_some() {
                "resource_link"
            } else {
                "other"
            }
        })
        .collect();

    Ok(crate::ToolResult {
        result: serde_json::json!({
            "content": content,
            "text": text,
            "structured_content": result.structured_content,
        }),
        // The agent path serializes a tool result into the prompt on every
        // later turn, and `content`'s text blocks are the same bytes as `text`
        // — so sending both charges twice for one message. Drop `content` for
        // the model and keep it in `result` for workflow templates, which are
        // rendered once and may need image or resource blocks.
        //
        // Non-text blocks are summarized rather than dropped silently: a model
        // told only the text would otherwise have no idea an image came back.
        for_model: Some(serde_json::json!({
            "text": text,
            "structured_content": result.structured_content,
            "non_text_blocks": non_text_kinds,
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{CallToolResult, ContentBlock};
    use serde_json::json;

    fn result_with(content: Vec<ContentBlock>) -> CallToolResult {
        // `CallToolResult` is `#[non_exhaustive]` in this rmcp version, so no
        // struct-literal expression (even with `..Default::default()`) compiles
        // from outside the rmcp crate. Build via `Default::default()` and assign
        // the one field we care about — same `None`/`None` values the brief's
        // literal spelled out explicitly.
        let mut result = CallToolResult::default();
        result.content = content;
        result
    }

    #[test]
    fn single_text_block_populates_text_and_content() {
        let mapped = map_call_result("t", result_with(vec![ContentBlock::text("hello")]))
            .expect("mapping succeeds");
        assert_eq!(mapped.result["text"], "hello");
        // Content blocks are passed through verbatim in MCP's own wire shape.
        assert_eq!(mapped.result["content"][0]["type"], "text");
        assert_eq!(mapped.result["content"][0]["text"], "hello");
        assert!(mapped.result["structured_content"].is_null());
    }

    #[test]
    fn for_model_drops_content_but_keeps_text_and_structured() {
        let mut raw = result_with(vec![ContentBlock::text("hello")]);
        raw.structured_content = Some(json!({ "answer": 42 }));
        let mapped = map_call_result("t", raw).expect("mapping succeeds");

        let fm = mapped.for_model.as_ref().expect("for_model is set for MCP");
        let mut keys: Vec<&str> = fm
            .as_object()
            .expect("for_model is an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort();
        assert_eq!(keys, vec!["non_text_blocks", "structured_content", "text"]);
        assert_eq!(fm["text"], "hello");
        assert_eq!(fm["structured_content"]["answer"], 42);
        // The whole point: the duplicated content blocks are gone from the
        // value the model sees, while `result` still carries them.
        assert!(
            fm.get("content").is_none(),
            "content must not be duplicated"
        );
        assert_eq!(mapped.result["content"][0]["text"], "hello");
    }

    #[test]
    fn for_model_names_non_text_blocks_without_their_payloads() {
        let raw = result_with(vec![
            ContentBlock::image("aGk=".to_string(), "image/png".to_string()),
            ContentBlock::text("caption"),
        ]);
        let mapped = map_call_result("t", raw).expect("mapping succeeds");
        let fm = mapped.for_model.as_ref().expect("for_model is set");
        assert_eq!(fm["non_text_blocks"], json!(["image"]));
        assert_eq!(fm["text"], "caption");
        // The base64 payload would dwarf the prompt; it stays in `result` only.
        assert!(
            !fm.to_string().contains("aGk="),
            "image payload must not reach the model projection: {fm}"
        );
        assert_eq!(mapped.result["content"][0]["data"], "aGk=");
    }

    #[test]
    fn for_model_reports_no_non_text_blocks_for_a_text_only_result() {
        let mapped = map_call_result("t", result_with(vec![ContentBlock::text("hi")]))
            .expect("mapping succeeds");
        let fm = mapped.for_model.as_ref().expect("for_model is set");
        assert_eq!(fm["non_text_blocks"], json!([]));
    }

    #[test]
    fn several_text_blocks_join_with_newlines() {
        let mapped = map_call_result(
            "t",
            result_with(vec![ContentBlock::text("one"), ContentBlock::text("two")]),
        )
        .expect("mapping succeeds");
        assert_eq!(mapped.result["text"], "one\ntwo");
        assert_eq!(mapped.result["content"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn empty_content_yields_empty_text_and_empty_array() {
        let mapped = map_call_result("t", result_with(vec![])).expect("mapping succeeds");
        assert_eq!(mapped.result["text"], "");
        assert_eq!(mapped.result["content"], json!([]));
        assert!(mapped.result["structured_content"].is_null());
    }

    #[test]
    fn structured_content_is_passed_through_verbatim() {
        let mut raw = result_with(vec![ContentBlock::text("see structured")]);
        raw.structured_content = Some(json!({ "answer": 42 }));
        let mapped = map_call_result("t", raw).expect("mapping succeeds");
        assert_eq!(mapped.result["structured_content"]["answer"], 42);
        // text stays populated even when structured output is present.
        assert_eq!(mapped.result["text"], "see structured");
    }

    #[test]
    fn all_three_keys_are_always_present() {
        let mapped = map_call_result("t", result_with(vec![])).expect("mapping succeeds");
        let obj = mapped.result.as_object().expect("result is an object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(keys, vec!["content", "structured_content", "text"]);
    }

    #[test]
    fn text_that_looks_like_json_stays_a_string() {
        let mapped = map_call_result("t", result_with(vec![ContentBlock::text("123")]))
            .expect("mapping succeeds");
        assert!(
            mapped.result["text"].is_string(),
            "must not become a number"
        );
        assert_eq!(mapped.result["text"], "123");

        let mapped = map_call_result("t", result_with(vec![ContentBlock::text("true")]))
            .expect("mapping succeeds");
        assert!(
            mapped.result["text"].is_string(),
            "must not become a boolean"
        );
        assert_eq!(mapped.result["text"], "true");
    }

    #[test]
    fn non_text_blocks_survive_in_content_but_not_text() {
        let raw = result_with(vec![
            ContentBlock::image("aGk=".to_string(), "image/png".to_string()),
            ContentBlock::text("caption"),
        ]);
        let mapped = map_call_result("t", raw).expect("mapping succeeds");
        assert_eq!(
            mapped.result["text"], "caption",
            "only text blocks join into text"
        );
        assert_eq!(mapped.result["content"].as_array().map(Vec::len), Some(2));
        // The image block must survive intact in MCP's own wire shape — this is
        // rmcp's serde representation, not Flux's own type, so an rmcp bump
        // renaming a field would otherwise change Flux's public output silently.
        assert_eq!(mapped.result["content"][0]["type"], "image");
        assert_eq!(mapped.result["content"][0]["data"], "aGk=");
        assert_eq!(mapped.result["content"][0]["mimeType"], "image/png");
    }

    #[test]
    fn is_error_becomes_a_provider_error_carrying_the_text() {
        let mut raw = result_with(vec![ContentBlock::text("boom")]);
        raw.is_error = Some(true);
        match map_call_result("blow_up", raw) {
            Err(FluxError::Provider(m)) => {
                assert!(m.contains("blow_up"), "names the tool: {m}");
                assert!(m.contains("boom"), "carries the message: {m}");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[test]
    fn is_error_with_no_text_still_errors() {
        let mut raw = result_with(vec![]);
        raw.is_error = Some(true);
        match map_call_result("blow_up", raw) {
            Err(FluxError::Provider(_)) => {}
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[test]
    fn is_error_false_is_a_success() {
        let mut raw = result_with(vec![ContentBlock::text("fine")]);
        raw.is_error = Some(false);
        assert!(map_call_result("t", raw).is_ok());
    }

    #[test]
    fn object_args_convert_and_null_args_become_none() {
        assert!(matches!(
            arguments_for("t", &json!({ "a": 1 })),
            Ok(Some(_))
        ));
        assert!(matches!(arguments_for("t", &json!(null)), Ok(None)));
    }

    #[tokio::test]
    async fn within_turns_a_silent_server_into_a_provider_error() {
        // The failure being guarded: a child that is alive but never answers.
        // `pending()` is exactly that — no error, no close, no EOF.
        let budget = Duration::from_millis(10);
        let hangs = std::future::pending::<Result<()>>();
        match within(budget, "tools/call", "my-server args", hangs).await {
            Err(FluxError::Provider(m)) => {
                assert!(m.contains("my-server args"), "names the server: {m}");
                assert!(m.contains("tools/call"), "names the request: {m}");
                // Names the budget, so an operator can tell a timeout from a
                // server-side error without reading the source.
                assert!(m.contains('0'), "names the budget: {m}");
            }
            other => panic!("expected a Provider timeout error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn within_is_a_passthrough_when_the_call_answers() {
        let got = within(
            Duration::from_secs(30),
            "tools/list",
            "s",
            std::future::ready(Ok(7)),
        )
        .await;
        assert_eq!(got.expect("passes the value through"), 7);
    }

    #[tokio::test]
    async fn within_does_not_disturb_the_calls_own_error() {
        // A real server error must arrive unchanged — reporting it as a timeout
        // would send an operator looking for a hang that never happened.
        let inner = std::future::ready(Err::<(), _>(FluxError::Provider(
            "listing tools from MCP server `s`: boom".into(),
        )));
        match within(Duration::from_secs(30), "tools/list", "s", inner).await {
            Err(FluxError::Provider(m)) => {
                assert!(m.contains("boom"), "keeps the original message: {m}");
                assert!(!m.contains("within"), "not reframed as a timeout: {m}");
            }
            other => panic!("expected the inner Provider error, got {other:?}"),
        }
    }

    #[test]
    fn call_budget_exceeds_the_boot_budgets() {
        // A tool doing real work must not be held to a boot-time budget.
        assert!(CALL_TOOL_TIMEOUT > LIST_TOOLS_TIMEOUT);
        assert!(CALL_TOOL_TIMEOUT > HANDSHAKE_TIMEOUT);
    }

    #[test]
    fn non_object_args_are_a_config_error() {
        match arguments_for("weather", &json!(["not", "an", "object"])) {
            Err(FluxError::Config(m)) => assert!(m.contains("weather"), "names the tool: {m}"),
            other => panic!("expected Config error, got {other:?}"),
        }
        assert!(arguments_for("weather", &json!("bare string")).is_err());
        assert!(arguments_for("weather", &json!(7)).is_err());
    }
}
