// Portions adapted from rig (MIT), commit 55cd481a — see NOTICE
//! Anthropic Messages API backend (streaming).

use crate::http::{self, parse_tool_args, sse_stream, token_count, SseDecoder};
use crate::types::{
    ChatRequest, Content, LlmBackend, LlmError, LlmEvent, LlmStream, Message, OutputFormat,
    ProviderBlock, Role, Usage,
};
use futures_util::future::BoxFuture;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const PROVIDER: &str = "anthropic";

pub struct AnthropicBackend {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    api_key_env: Option<String>,
}

impl AnthropicBackend {
    pub fn new(
        http: reqwest::Client,
        base_url: String,
        api_key: Option<String>,
        api_key_env: Option<String>,
    ) -> Self {
        AnthropicBackend {
            http,
            base_url,
            api_key,
            api_key_env,
        }
    }

    fn messages_url(&self) -> String {
        format!("{}/messages", self.base_url.trim_end_matches('/'))
    }
}

impl LlmBackend for AnthropicBackend {
    fn stream(&self, req: ChatRequest) -> BoxFuture<'_, Result<LlmStream, LlmError>> {
        Box::pin(async move {
            let Some(key) = self.api_key.as_deref() else {
                return Err(LlmError::Config(match self.api_key_env.as_deref() {
                    Some(var) => format!("{var} is not set"),
                    None => "no API key configured for this anthropic provider \
                             (set api_key_env)"
                        .into(),
                }));
            };
            let body = request_body(&req)?;
            let resp = self
                .http
                .post(self.messages_url())
                .header("x-api-key", key)
                .header("anthropic-version", API_VERSION)
                .json(&body)
                .send()
                .await
                .map_err(http::transport_error)?;
            let resp = http::error_for_status(resp).await?;
            Ok(sse_stream(resp, Decoder::default()))
        })
    }
}

fn request_body(req: &ChatRequest) -> Result<Value, LlmError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(req.model));
    // Anthropic requires the thinking budget to be below max_tokens.
    let default_max = match req.reasoning.as_ref().and_then(|r| r.budget_tokens) {
        Some(b) => DEFAULT_MAX_TOKENS.saturating_add(u32::try_from(b).unwrap_or(0)),
        None => DEFAULT_MAX_TOKENS,
    };
    body.insert(
        "max_tokens".into(),
        json!(req.max_tokens.unwrap_or(default_max)),
    );
    body.insert("stream".into(), json!(true));
    if !req.system.is_empty() {
        body.insert(
            "system".into(),
            json!([{"type": "text", "text": req.system}]),
        );
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    body.insert("messages".into(), json!(map_messages(&req.messages)));
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({"name": t.name, "description": t.description, "input_schema": t.parameters})
            })
            .collect();
        body.insert("tools".into(), json!(tools));
    }

    let mut output_config = Map::new();
    if let OutputFormat::Json { schema } = &req.output {
        output_config.insert(
            "format".into(),
            json!({"type": "json_schema", "schema": schema}),
        );
    }
    if let Some(r) = &req.reasoning {
        match (&r.effort, r.budget_tokens) {
            (Some(_), Some(_)) => {
                return Err(LlmError::Config(
                    "cannot set both reasoning_effort and thinking_budget_tokens for anthropic"
                        .into(),
                ))
            }
            (Some(effort), None) => {
                body.insert("thinking".into(), json!({"type": "adaptive"}));
                output_config.insert("effort".into(), json!(effort));
            }
            (None, Some(budget)) => {
                body.insert(
                    "thinking".into(),
                    json!({"type": "enabled", "budget_tokens": budget}),
                );
            }
            (None, None) => {}
        }
    }
    if !output_config.is_empty() {
        body.insert("output_config".into(), Value::Object(output_config));
    }
    Ok(Value::Object(body))
}

/// Anthropic needs every `tool_result` of a turn in one user message, and
/// rejects consecutive same-role messages, so adjacent ones are merged.
fn map_messages(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<(Role, Vec<Value>)> = Vec::new();
    for m in messages {
        let blocks = map_content(&m.content);
        match out.last_mut() {
            Some((role, existing)) if *role == m.role => existing.extend(blocks),
            _ => out.push((m.role, blocks)),
        }
    }
    out.into_iter()
        .map(|(role, content)| {
            let role = match role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            json!({"role": role, "content": content})
        })
        .collect()
}

/// Opaque (thinking) blocks must come first in an assistant turn.
fn map_content(content: &[Content]) -> Vec<Value> {
    let mut opaque = Vec::new();
    let mut rest = Vec::new();
    for c in content {
        match c {
            Content::Opaque(b) if b.provider == PROVIDER => opaque.push(b.data.clone()),
            Content::Opaque(_) => {}
            Content::Text(text) => rest.push(json!({"type": "text", "text": text})),
            Content::ToolCall { id, name, args } => {
                rest.push(json!({"type": "tool_use", "id": id, "name": name, "input": args}))
            }
            Content::ToolResult { id, result, .. } => rest.push(
                json!({"type": "tool_result", "tool_use_id": id, "content": result.to_string()}),
            ),
        }
    }
    opaque.extend(rest);
    opaque
}

/// A content block being assembled from stream deltas, keyed by `index`.
enum Block {
    Text,
    Thinking {
        thinking: String,
        signature: String,
    },
    Redacted(Value),
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    Other,
}

#[derive(Default)]
struct Decoder {
    blocks: HashMap<u64, Block>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    model: Option<String>,
    done: bool,
}

impl SseDecoder for Decoder {
    fn done(&self) -> bool {
        self.done
    }

    /// Handles one SSE event; returns the events it produces.
    fn handle(&mut self, name: &str, data: &str) -> Result<Vec<LlmEvent>, LlmError> {
        if data.trim().is_empty() {
            return Ok(vec![]);
        }
        let v = serde_json::from_str::<Value>(data).map_err(|e| LlmError::Stream {
            message: format!("malformed SSE data from provider: {e}"),
            retryable: true,
        })?;
        let kind = v["type"].as_str().unwrap_or(name);
        let mut out = Vec::new();
        match kind {
            "message_start" => {
                let m = &v["message"];
                self.model = m["model"].as_str().map(str::to_owned);
                self.input_tokens = token_count(&m["usage"]["input_tokens"]);
                self.output_tokens = token_count(&m["usage"]["output_tokens"]);
            }
            "content_block_start" => {
                let index = v["index"].as_u64().unwrap_or_default();
                let b = &v["content_block"];
                let block = match b["type"].as_str() {
                    Some("text") => {
                        if let Some(t) = b["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push(LlmEvent::Text(t.to_owned()));
                        }
                        Block::Text
                    }
                    Some("thinking") => {
                        let thinking = b["thinking"].as_str().unwrap_or_default().to_owned();
                        if !thinking.is_empty() {
                            out.push(LlmEvent::Thought(thinking.clone()));
                        }
                        Block::Thinking {
                            thinking,
                            signature: b["signature"].as_str().unwrap_or_default().to_owned(),
                        }
                    }
                    Some("redacted_thinking") => Block::Redacted(b.clone()),
                    Some("tool_use") => Block::ToolUse {
                        id: b["id"].as_str().unwrap_or_default().to_owned(),
                        name: b["name"].as_str().unwrap_or_default().to_owned(),
                        json: String::new(),
                    },
                    _ => Block::Other,
                };
                self.blocks.insert(index, block);
            }
            "content_block_delta" => {
                let index = v["index"].as_u64().unwrap_or_default();
                let d = &v["delta"];
                match (d["type"].as_str(), self.blocks.get_mut(&index)) {
                    (Some("text_delta"), _) => {
                        if let Some(t) = d["text"].as_str() {
                            out.push(LlmEvent::Text(t.to_owned()));
                        }
                    }
                    (Some("thinking_delta"), Some(Block::Thinking { thinking, .. })) => {
                        if let Some(t) = d["thinking"].as_str() {
                            thinking.push_str(t);
                            out.push(LlmEvent::Thought(t.to_owned()));
                        }
                    }
                    (Some("signature_delta"), Some(Block::Thinking { signature, .. })) => {
                        signature.push_str(d["signature"].as_str().unwrap_or_default());
                    }
                    (Some("input_json_delta"), Some(Block::ToolUse { json, .. })) => {
                        json.push_str(d["partial_json"].as_str().unwrap_or_default());
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = v["index"].as_u64().unwrap_or_default();
                match self.blocks.remove(&index) {
                    Some(Block::ToolUse { id, name, json }) => {
                        out.push(parse_tool_args(&id, &name, &json))
                    }
                    Some(Block::Thinking {
                        thinking,
                        signature,
                    }) if !signature.is_empty() => out.push(LlmEvent::Opaque(ProviderBlock {
                        provider: PROVIDER.into(),
                        data: json!({"type": "thinking", "thinking": thinking,
                                     "signature": signature}),
                    })),
                    Some(Block::Redacted(block)) if block["data"].is_string() => {
                        out.push(LlmEvent::Opaque(ProviderBlock {
                            provider: PROVIDER.into(),
                            data: block,
                        }))
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(n) = token_count(&v["usage"]["output_tokens"]) {
                    self.output_tokens = Some(n);
                }
                match v["delta"]["stop_reason"].as_str() {
                    Some("refusal") => {
                        return Err(LlmError::Refused("the model refused to respond".into()))
                    }
                    Some("max_tokens") => {
                        tracing::warn!("model output was cut off at max_tokens")
                    }
                    Some("pause_turn") => {
                        return Err(LlmError::Stream {
                            message: "pause_turn is not supported".into(),
                            retryable: false,
                        })
                    }
                    _ => {}
                }
            }
            "message_stop" => {
                self.done = true;
                out.push(LlmEvent::Usage(Usage {
                    input_tokens: self.input_tokens,
                    output_tokens: self.output_tokens,
                    model: self.model.clone(),
                }));
            }
            "error" => {
                let kind = v["error"]["type"].as_str().unwrap_or("unknown_error");
                let message = v["error"]["message"].as_str().unwrap_or_default();
                return Err(LlmError::Stream {
                    message: format!("{kind}: {message}"),
                    retryable: matches!(kind, "overloaded_error" | "api_error"),
                });
            }
            _ => {}
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{serve, serve_stall};
    use crate::types::{
        ChatRequest, Content, LlmBackend, LlmError, LlmEvent, Message, OutputFormat, ProviderBlock,
        Reasoning, Role, ToolDef, Usage,
    };
    use crate::{stream_chat_with_idle, RetryConfig};
    use futures_util::StreamExt;
    use serde_json::json;
    use std::time::Duration;

    const TEXT: &str = include_str!("fixtures/anthropic/text.sse");
    const THINKING_TOOL: &str = include_str!("fixtures/anthropic/thinking_tool.sse");
    const PARALLEL: &str = include_str!("fixtures/anthropic/parallel_tools.sse");
    const MALFORMED: &str = include_str!("fixtures/anthropic/malformed_args.sse");
    const REFUSAL: &str = include_str!("fixtures/anthropic/refusal.sse");
    const TRUNCATED: &str = include_str!("fixtures/anthropic/truncated.sse");
    const GARBLED: &str = include_str!("fixtures/anthropic/garbled_data.sse");
    const OVERLOADED: &str = include_str!("fixtures/anthropic/error_overloaded.sse");

    fn req() -> ChatRequest {
        ChatRequest {
            model: "claude-test".into(),
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text("hi".into())],
            }],
            tools: vec![],
            output: OutputFormat::Text,
            max_tokens: None,
            temperature: None,
            reasoning: None,
        }
    }

    fn msg(role: Role, content: Vec<Content>) -> Message {
        Message { role, content }
    }

    fn tool_result(id: &str) -> Message {
        msg(
            Role::User,
            vec![Content::ToolResult {
                id: id.into(),
                name: "t".into(),
                result: json!({"ok": id}),
            }],
        )
    }

    fn backend(url: String) -> AnthropicBackend {
        AnthropicBackend::new(
            reqwest::Client::new(),
            url,
            Some("test-key".into()),
            Some("ANTHROPIC_API_KEY".into()),
        )
    }

    async fn collect(url: String) -> Vec<Result<LlmEvent, LlmError>> {
        let stream = backend(url).stream(req()).await.expect("stream opens");
        stream.collect().await
    }

    fn ok_events(items: Vec<Result<LlmEvent, LlmError>>) -> Vec<LlmEvent> {
        items.into_iter().map(|r| r.expect("no error")).collect()
    }

    #[test]
    fn request_maps_messages_tools_and_system() {
        let mut r = req();
        r.system = "be brief".into();
        r.temperature = Some(0.5);
        r.max_tokens = Some(100);
        r.messages.push(msg(
            Role::Assistant,
            vec![Content::ToolCall {
                id: "c1".into(),
                name: "get".into(),
                args: json!({"a": 1}),
            }],
        ));
        r.messages.push(tool_result("c1"));
        r.tools = vec![ToolDef {
            name: "get".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
        }];
        let b = request_body(&r).unwrap();
        assert_eq!(b["model"], "claude-test");
        assert_eq!(b["max_tokens"], 100);
        assert_eq!(b["stream"], true);
        assert_eq!(b["temperature"], 0.5);
        assert_eq!(b["system"], json!([{"type": "text", "text": "be brief"}]));
        assert_eq!(
            b["messages"][0],
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        );
        assert_eq!(
            b["messages"][1],
            json!({"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "get", "input": {"a": 1}}]})
        );
        assert_eq!(
            b["messages"][2]["content"][0],
            json!({"type": "tool_result", "tool_use_id": "c1", "content": "{\"ok\":\"c1\"}"})
        );
        assert_eq!(
            b["tools"],
            json!([{"name": "get", "description": "d", "input_schema": {"type": "object"}}])
        );
    }

    #[test]
    fn defaults_omit_empty_fields() {
        let b = request_body(&req()).unwrap();
        assert_eq!(b["max_tokens"], 4096);
        assert!(b.get("system").is_none());
        assert!(b.get("tools").is_none());
        assert!(b.get("temperature").is_none());
        assert!(b.get("thinking").is_none());
        assert!(b.get("output_config").is_none());
    }

    #[test]
    fn consecutive_tool_results_are_merged_into_one_user_message() {
        let mut r = req();
        r.messages = vec![tool_result("a"), tool_result("b")];
        let b = request_body(&r).unwrap();
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[0]["content"][1]["tool_use_id"], "b");
    }

    #[test]
    fn opaque_thinking_is_replayed_first_and_foreign_opaque_dropped() {
        let thinking = json!({"type": "thinking", "thinking": "hm", "signature": "s"});
        let mut r = req();
        r.messages = vec![msg(
            Role::Assistant,
            vec![
                Content::Text("answer".into()),
                Content::Opaque(ProviderBlock {
                    provider: "openai".into(),
                    data: json!({"foreign": true}),
                }),
                Content::Opaque(ProviderBlock {
                    provider: "anthropic".into(),
                    data: thinking.clone(),
                }),
            ],
        )];
        let b = request_body(&r).unwrap();
        let content = b["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0], thinking);
        assert_eq!(content[1]["type"], "text");
    }

    #[test]
    fn json_output_sets_output_config_format() {
        let mut r = req();
        r.output = OutputFormat::Json {
            schema: json!({"type": "object"}),
        };
        let b = request_body(&r).unwrap();
        assert_eq!(
            b["output_config"],
            json!({"format": {"type": "json_schema", "schema": {"type": "object"}}})
        );
    }

    #[test]
    fn effort_sets_adaptive_thinking_and_merges_into_output_config() {
        let mut r = req();
        r.output = OutputFormat::Json {
            schema: json!({"type": "object"}),
        };
        r.reasoning = Some(Reasoning {
            effort: Some("high".into()),
            budget_tokens: None,
        });
        let b = request_body(&r).unwrap();
        assert_eq!(b["thinking"], json!({"type": "adaptive"}));
        assert_eq!(b["output_config"]["effort"], "high");
        assert_eq!(b["output_config"]["format"]["type"], "json_schema");
    }

    #[test]
    fn budget_sets_enabled_thinking() {
        let mut r = req();
        r.reasoning = Some(Reasoning {
            effort: None,
            budget_tokens: Some(2048),
        });
        let b = request_body(&r).unwrap();
        assert_eq!(
            b["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048})
        );
    }

    #[test]
    fn effort_and_budget_together_is_a_config_error() {
        let mut r = req();
        r.reasoning = Some(Reasoning {
            effort: Some("low".into()),
            budget_tokens: Some(1024),
        });
        assert_eq!(
            request_body(&r).unwrap_err(),
            LlmError::Config(
                "cannot set both reasoning_effort and thinking_budget_tokens for anthropic".into()
            )
        );
    }

    #[tokio::test]
    async fn streams_text_and_usage() {
        let served = serve(200, vec![], TEXT);
        let events = ok_events(collect(served.url.clone()).await);
        assert_eq!(
            events,
            vec![
                LlmEvent::Text("Hello".into()),
                LlmEvent::Text(" world".into()),
                LlmEvent::Usage(Usage {
                    input_tokens: Some(12),
                    output_tokens: Some(7),
                    model: Some("claude-test".into()),
                }),
            ]
        );
        let body = served.last_body.lock().unwrap().clone().unwrap();
        assert_eq!(body["stream"], true);
    }

    #[tokio::test]
    async fn thinking_then_tool_call_yields_thought_opaque_and_tool_call() {
        let events = ok_events(collect(serve(200, vec![], THINKING_TOOL).url).await);
        assert_eq!(
            events[..3],
            [
                LlmEvent::Thought("Need the weather.".into()),
                LlmEvent::Opaque(ProviderBlock {
                    provider: "anthropic".into(),
                    data: json!({"type": "thinking", "thinking": "Need the weather.",
                                 "signature": "sig-abc"}),
                }),
                LlmEvent::ToolCall {
                    id: "toolu_1".into(),
                    name: "get_weather".into(),
                    args: json!({"city": "Paris"}),
                },
            ]
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[3], LlmEvent::Usage(u) if u.output_tokens == Some(25)));
    }

    #[tokio::test]
    async fn parallel_tool_calls_are_assembled_per_block_index() {
        let events = ok_events(collect(serve(200, vec![], PARALLEL).url).await);
        assert_eq!(
            events[..2],
            [
                LlmEvent::ToolCall {
                    id: "toolu_a".into(),
                    name: "alpha".into(),
                    args: json!({"x": 1}),
                },
                LlmEvent::ToolCall {
                    id: "toolu_b".into(),
                    name: "beta".into(),
                    args: json!({"y": 2}),
                },
            ]
        );
    }

    #[tokio::test]
    async fn malformed_tool_json_becomes_invalid_tool_call() {
        let events = ok_events(collect(serve(200, vec![], MALFORMED).url).await);
        assert!(matches!(
            &events[0],
            LlmEvent::InvalidToolCall { id, name, .. } if id == "toolu_bad" && name == "broken"
        ));
    }

    #[tokio::test]
    async fn refusal_is_an_error() {
        let items = collect(serve(200, vec![], REFUSAL).url).await;
        assert!(matches!(items.last(), Some(Err(LlmError::Refused(_)))));
        assert!(!items.iter().any(|i| matches!(i, Ok(LlmEvent::Usage(_)))));
    }

    #[tokio::test]
    async fn early_end_is_a_retryable_error() {
        let items = collect(serve(200, vec![], TRUNCATED).url).await;
        assert_eq!(items[0], Ok(LlmEvent::Text("partial".into())));
        match items.last() {
            Some(Err(e @ LlmError::Stream { message, .. })) => {
                assert!(e.is_retryable());
                assert_eq!(message, "stream ended early");
            }
            other => panic!("expected stream error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn overloaded_stream_error_is_retryable() {
        let items = collect(serve(200, vec![], OVERLOADED).url).await;
        match items.last() {
            Some(Err(e @ LlmError::Stream { .. })) => assert!(e.is_retryable()),
            other => panic!("expected stream error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_json_sse_data_is_a_stream_error() {
        let items = collect(serve(200, vec![], GARBLED).url).await;
        match items.last() {
            Some(Err(e @ LlmError::Stream { message, .. })) => {
                assert!(e.is_retryable());
                assert!(message.starts_with("malformed SSE data from provider: "));
                assert!(!message.contains("not-json"));
            }
            other => panic!("expected stream error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn status_429_carries_retry_after() {
        let served = serve(
            429,
            vec![("retry-after", "3".to_string())],
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
        );
        match backend(served.url).stream(req()).await {
            Err(LlmError::Http {
                status,
                retry_after,
                message,
            }) => {
                assert_eq!(status, 429);
                assert_eq!(retry_after, Some(Duration::from_secs(3)));
                assert_eq!(message, "slow down");
            }
            other => panic!("expected Http error, got {:?}", other.err()),
        }
    }

    #[tokio::test]
    async fn status_401_is_not_retryable() {
        let served = serve(401, vec![], "unauthorized");
        let err = backend(served.url).stream(req()).await.err().unwrap();
        assert!(matches!(err, LlmError::Http { status: 401, .. }));
        assert!(!err.is_retryable());
    }

    #[tokio::test]
    async fn missing_key_without_env_name_does_not_name_a_variable() {
        let served = serve(200, vec![], TEXT);
        let b = AnthropicBackend::new(reqwest::Client::new(), served.url.clone(), None, None);
        let err = b.stream(req()).await.err().unwrap();
        match err {
            LlmError::Config(m) => {
                assert!(!m.contains("ANTHROPIC_API_KEY"), "{m}");
                assert!(m.contains("api_key_env"), "{m}");
            }
            other => panic!("expected config error, got {other:?}"),
        }
        assert!(served.last_body.lock().unwrap().is_none());
    }

    #[test]
    fn default_max_tokens_exceeds_the_thinking_budget() {
        let mut r = req();
        r.reasoning = Some(Reasoning {
            effort: None,
            budget_tokens: Some(8000),
        });
        let b = request_body(&r).unwrap();
        assert_eq!(b["max_tokens"], 8000 + 4096);
        assert_eq!(request_body(&req()).unwrap()["max_tokens"], 4096);
        r.max_tokens = Some(20000);
        assert_eq!(request_body(&r).unwrap()["max_tokens"], 20000);
    }

    #[tokio::test]
    async fn missing_key_names_the_env_var_only() {
        let served = serve(200, vec![], TEXT);
        let b = AnthropicBackend::new(
            reqwest::Client::new(),
            served.url.clone(),
            None,
            Some("ANTHROPIC_API_KEY".into()),
        );
        let err = b.stream(req()).await.err().unwrap();
        assert_eq!(err, LlmError::Config("ANTHROPIC_API_KEY is not set".into()));
        assert!(served.last_body.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn stalled_connection_fails_through_stream_chat() {
        let b = backend(serve_stall());
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let retry = RetryConfig {
            max_attempts: 1,
            base_delay_ms: 1,
        };
        let err = stream_chat_with_idle(&b, req(), &tx, retry, Duration::from_millis(50))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("stalled"), "{err}");
    }
}
