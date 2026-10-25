// Portions adapted from rig (MIT), commit 55cd481a — see NOTICE
//! OpenAI-compatible Chat Completions backend (streaming).

use crate::http::{self, parse_tool_args, sse_stream, token_count, SseDecoder};
use crate::types::{
    ChatRequest, Content, LlmBackend, LlmError, LlmEvent, LlmStream, Message, OutputFormat, Role,
    Usage,
};
use futures_util::future::BoxFuture;
use serde_json::{json, Map, Value};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAiBackend {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    api_key_env: Option<String>,
}

impl OpenAiBackend {
    /// A provider with an `api_key_env` requires that key: `stream` fails with
    /// a `Config` error naming the variable when `api_key` is `None`. A provider
    /// without one sends no `Authorization` header (local servers).
    pub fn new(
        http: reqwest::Client,
        base_url: String,
        api_key: Option<String>,
        api_key_env: Option<String>,
    ) -> Self {
        OpenAiBackend {
            http,
            base_url,
            api_key,
            api_key_env,
        }
    }

    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

impl LlmBackend for OpenAiBackend {
    fn stream(&self, req: ChatRequest) -> BoxFuture<'_, Result<LlmStream, LlmError>> {
        Box::pin(async move {
            if self.api_key.is_none() {
                if let Some(var) = self.api_key_env.as_deref() {
                    return Err(LlmError::Config(format!("{var} is not set")));
                }
            }
            let mut builder = self
                .http
                .post(self.completions_url())
                .json(&request_body(&req));
            if let Some(key) = self.api_key.as_deref() {
                builder = builder.bearer_auth(key);
            }
            let resp = builder.send().await.map_err(http::transport_error)?;
            let resp = http::error_for_status(resp).await?;
            Ok(sse_stream(resp, Decoder::default()))
        })
    }
}

fn request_body(req: &ChatRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(req.model));
    body.insert("stream".into(), json!(true));
    body.insert("stream_options".into(), json!({"include_usage": true}));
    body.insert(
        "messages".into(),
        json!(map_messages(&req.system, &req.messages)),
    );
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters}})
            })
            .collect();
        body.insert("tools".into(), json!(tools));
    }
    if let Some(n) = req.max_tokens {
        body.insert("max_completion_tokens".into(), json!(n));
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let OutputFormat::Json { schema } = &req.output {
        body.insert(
            "response_format".into(),
            json!({"type": "json_schema", "json_schema": {
                "name": "response", "strict": true, "schema": schema}}),
        );
    }
    if let Some(r) = &req.reasoning {
        if let Some(effort) = &r.effort {
            body.insert("reasoning_effort".into(), json!(effort));
        }
        if r.budget_tokens.is_some() {
            tracing::warn!("thinking_budget_tokens is ignored for openai; use reasoning_effort");
        }
    }
    Value::Object(body)
}

fn map_messages(system: &str, messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }
    for m in messages {
        let mut texts: Vec<&str> = Vec::new();
        let mut calls: Vec<Value> = Vec::new();
        let mut results: Vec<Value> = Vec::new();
        for c in &m.content {
            match c {
                Content::Text(t) => texts.push(t),
                Content::ToolCall { id, name, args } => calls.push(json!({
                    "id": id, "type": "function",
                    "function": {"name": name, "arguments": args.to_string()}})),
                Content::ToolResult { id, result, .. } => results.push(
                    json!({"role": "tool", "tool_call_id": id, "content": result.to_string()}),
                ),
                Content::Opaque(_) => {}
            }
        }
        match m.role {
            Role::User => {
                // Tool messages must directly follow the assistant's tool calls.
                out.extend(results);
                if !texts.is_empty() {
                    out.push(json!({"role": "user", "content": texts.join("\n")}));
                }
            }
            Role::Assistant => {
                let content = if texts.is_empty() {
                    Value::Null
                } else {
                    json!(texts.join("\n"))
                };
                let mut msg = json!({"role": "assistant", "content": content});
                if !calls.is_empty() {
                    msg["tool_calls"] = json!(calls);
                }
                out.push(msg);
            }
        }
    }
    out
}

/// A tool call being assembled from fragments.
struct PendingCall {
    index: Option<u64>,
    id: String,
    name: String,
    args: String,
}

#[derive(Default)]
struct Decoder {
    calls: Vec<PendingCall>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    model: Option<String>,
    done: bool,
}

impl SseDecoder for Decoder {
    fn done(&self) -> bool {
        self.done
    }

    /// Handles one SSE `data:` payload; returns the events it produces.
    fn handle(&mut self, _event: &str, data: &str) -> Result<Vec<LlmEvent>, LlmError> {
        let data = data.trim();
        if data.is_empty() {
            return Ok(vec![]);
        }
        if data == "[DONE]" {
            self.done = true;
            return Ok(self.finish());
        }
        let v = serde_json::from_str::<Value>(data).map_err(|e| LlmError::Stream {
            message: format!("malformed SSE data from provider: {e}"),
            retryable: true,
        })?;
        if !v["error"].is_null() {
            let message = v["error"]["message"]
                .as_str()
                .or_else(|| v["error"].as_str())
                .unwrap_or("provider returned an error");
            return Err(LlmError::Stream {
                message: message.to_owned(),
                retryable: false,
            });
        }
        if let Some(m) = v["model"].as_str() {
            self.model = Some(m.to_owned());
        }
        if let Some(n) = token_count(&v["usage"]["prompt_tokens"]) {
            self.input_tokens = Some(n);
        }
        if let Some(n) = token_count(&v["usage"]["completion_tokens"]) {
            self.output_tokens = Some(n);
        }
        if v["choices"][0]["finish_reason"].as_str() == Some("length") {
            tracing::warn!("model output was cut off at max_tokens");
        }
        let mut out = Vec::new();
        let delta = &v["choices"][0]["delta"];
        for field in ["reasoning_content", "reasoning", "reasoning_text"] {
            if let Some(t) = delta[field].as_str().filter(|t| !t.is_empty()) {
                out.push(LlmEvent::Thought(t.to_owned()));
                break;
            }
        }
        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            out.push(LlmEvent::Text(t.to_owned()));
        }
        if let Some(fragments) = delta["tool_calls"].as_array() {
            for f in fragments {
                self.add_fragment(f);
            }
        }
        Ok(out)
    }
}

impl Decoder {
    /// Matches a fragment by `index`, else by `id`, else the latest call.
    fn add_fragment(&mut self, f: &Value) {
        let index = f["index"].as_u64();
        let id = f["id"].as_str().filter(|s| !s.is_empty());
        let pos = match (index, id) {
            (Some(i), _) => self.calls.iter().position(|c| c.index == Some(i)),
            (None, Some(id)) => self.calls.iter().position(|c| c.id == id),
            (None, None) => self.calls.len().checked_sub(1),
        };
        let pos = pos.unwrap_or_else(|| {
            self.calls.push(PendingCall {
                index,
                id: String::new(),
                name: String::new(),
                args: String::new(),
            });
            self.calls.len() - 1
        });
        let call = &mut self.calls[pos];
        if call.id.is_empty() {
            if let Some(id) = id {
                call.id = id.to_owned();
            }
        }
        let func = &f["function"];
        if call.name.is_empty() {
            if let Some(n) = func["name"].as_str() {
                call.name = n.to_owned();
            }
        }
        if let Some(a) = func["arguments"].as_str() {
            call.args.push_str(a);
        }
    }

    /// Assembled calls in index order, then the single `Usage`.
    fn finish(&mut self) -> Vec<LlmEvent> {
        let mut calls = std::mem::take(&mut self.calls);
        calls.sort_by_key(|c| c.index.unwrap_or(u64::MAX));
        let mut out: Vec<LlmEvent> = calls
            .iter()
            .map(|c| parse_tool_args(&c.id, &c.name, &c.args))
            .collect();
        out.push(LlmEvent::Usage(Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            model: self.model.clone(),
        }));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::serve;
    use crate::types::{
        ChatRequest, Content, LlmBackend, LlmError, LlmEvent, Message, OutputFormat, ProviderBlock,
        Reasoning, Role, ToolDef, Usage,
    };
    use futures_util::StreamExt;
    use serde_json::json;

    const TEXT: &str = include_str!("fixtures/openai/text.sse");
    const REASONING: &str = include_str!("fixtures/openai/reasoning.sse");
    const PARALLEL: &str = include_str!("fixtures/openai/parallel_tools.sse");
    const MALFORMED: &str = include_str!("fixtures/openai/malformed_args.sse");
    const NO_USAGE: &str = include_str!("fixtures/openai/no_usage.sse");
    const TRUNCATED: &str = include_str!("fixtures/openai/truncated.sse");
    const ERROR_BODY: &str = include_str!("fixtures/openai/error_body.sse");
    const GARBLED: &str = include_str!("fixtures/openai/garbled_data.sse");

    fn req() -> ChatRequest {
        ChatRequest {
            model: "gpt-test".into(),
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

    fn backend(url: String, key: Option<&str>) -> OpenAiBackend {
        let env = key.map(|_| "OPENAI_API_KEY".to_owned());
        OpenAiBackend::new(reqwest::Client::new(), url, key.map(str::to_owned), env)
    }

    async fn collect(url: String) -> Vec<Result<LlmEvent, LlmError>> {
        let stream = backend(url, Some("test-key"))
            .stream(req())
            .await
            .expect("stream opens");
        stream.collect().await
    }

    fn ok_events(items: Vec<Result<LlmEvent, LlmError>>) -> Vec<LlmEvent> {
        items.into_iter().map(|r| r.expect("no error")).collect()
    }

    #[test]
    fn request_maps_roles_tool_calls_and_tool_results() {
        let mut r = req();
        r.system = "be brief".into();
        r.temperature = Some(0.5);
        r.max_tokens = Some(100);
        r.messages = vec![
            msg(
                Role::User,
                vec![Content::Text("a".into()), Content::Text("b".into())],
            ),
            msg(
                Role::Assistant,
                vec![
                    Content::Opaque(ProviderBlock {
                        provider: "anthropic".into(),
                        data: json!({"x": 1}),
                    }),
                    Content::ToolCall {
                        id: "c1".into(),
                        name: "get".into(),
                        args: json!({"a": 1}),
                    },
                    Content::ToolCall {
                        id: "c2".into(),
                        name: "put".into(),
                        args: json!({}),
                    },
                ],
            ),
            msg(
                Role::User,
                vec![Content::ToolResult {
                    id: "c1".into(),
                    name: "get".into(),
                    result: json!({"ok": 1}),
                }],
            ),
            msg(
                Role::User,
                vec![Content::ToolResult {
                    id: "c2".into(),
                    name: "put".into(),
                    result: json!("done"),
                }],
            ),
            msg(Role::Assistant, vec![Content::Text("final".into())]),
        ];
        r.tools = vec![ToolDef {
            name: "get".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
        }];
        let b = request_body(&r);
        assert_eq!(b["model"], "gpt-test");
        assert_eq!(b["stream"], true);
        assert_eq!(b["stream_options"], json!({"include_usage": true}));
        assert_eq!(b["max_completion_tokens"], 100);
        assert_eq!(b["temperature"], 0.5);
        let m = b["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "be brief"}));
        assert_eq!(m[1], json!({"role": "user", "content": "a\nb"}));
        assert_eq!(
            m[2],
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get", "arguments": "{\"a\":1}"}},
                {"id": "c2", "type": "function",
                 "function": {"name": "put", "arguments": "{}"}}]})
        );
        assert_eq!(
            m[3],
            json!({"role": "tool", "tool_call_id": "c1", "content": "{\"ok\":1}"})
        );
        assert_eq!(
            m[4],
            json!({"role": "tool", "tool_call_id": "c2", "content": "\"done\""})
        );
        assert_eq!(m[5], json!({"role": "assistant", "content": "final"}));
        assert_eq!(m.len(), 6);
        assert_eq!(
            b["tools"],
            json!([{"type": "function", "function": {
                "name": "get", "description": "d", "parameters": {"type": "object"}}}])
        );
    }

    #[test]
    fn defaults_omit_empty_fields() {
        let b = request_body(&req());
        assert!(b.get("tools").is_none());
        assert!(b.get("temperature").is_none());
        assert!(b.get("max_completion_tokens").is_none());
        assert!(b.get("response_format").is_none());
        assert!(b.get("reasoning_effort").is_none());
        assert_eq!(b["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn json_output_sets_strict_json_schema() {
        let mut r = req();
        r.output = OutputFormat::Json {
            schema: json!({"type": "object"}),
        };
        assert_eq!(
            request_body(&r)["response_format"],
            json!({"type": "json_schema", "json_schema": {
                "name": "response", "strict": true, "schema": {"type": "object"}}})
        );
    }

    #[test]
    fn effort_maps_to_reasoning_effort() {
        let mut r = req();
        r.reasoning = Some(Reasoning {
            effort: Some("high".into()),
            budget_tokens: Some(2048),
        });
        let b = request_body(&r);
        assert_eq!(b["reasoning_effort"], "high");
        assert!(b.get("thinking").is_none());
        assert!(b.get("budget_tokens").is_none());
    }

    #[test]
    fn base_url_trailing_slash_is_trimmed() {
        let b = backend("http://x/v1/".into(), None);
        assert_eq!(b.completions_url(), "http://x/v1/chat/completions");
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
                    model: Some("gpt-test".into()),
                }),
            ]
        );
        let body = served.last_body.lock().unwrap().clone().unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(
            served.last_path.lock().unwrap().as_deref(),
            Some("/chat/completions")
        );
        let headers = served.last_headers.lock().unwrap().clone().unwrap();
        assert_eq!(headers["authorization"], "Bearer test-key");
    }

    #[tokio::test]
    async fn reasoning_fields_become_thoughts() {
        let events = ok_events(collect(serve(200, vec![], REASONING).url).await);
        assert_eq!(
            events[..4],
            [
                LlmEvent::Thought("think1 ".into()),
                LlmEvent::Thought("think2 ".into()),
                LlmEvent::Thought("think3".into()),
                LlmEvent::Text("done".into()),
            ]
        );
    }

    #[tokio::test]
    async fn interleaved_parallel_tool_calls_are_assembled_by_index() {
        let events = ok_events(collect(serve(200, vec![], PARALLEL).url).await);
        assert_eq!(
            events[..2],
            [
                LlmEvent::ToolCall {
                    id: "call_a".into(),
                    name: "alpha".into(),
                    args: json!({"x": 1}),
                },
                LlmEvent::ToolCall {
                    id: "call_b".into(),
                    name: "beta".into(),
                    args: json!({"y": 2}),
                },
            ]
        );
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[2], LlmEvent::Usage(u) if u.output_tokens == Some(5)));
    }

    #[tokio::test]
    async fn malformed_tool_json_becomes_invalid_tool_call() {
        let events = ok_events(collect(serve(200, vec![], MALFORMED).url).await);
        assert!(matches!(
            &events[0],
            LlmEvent::InvalidToolCall { id, name, .. } if id == "call_bad" && name == "broken"
        ));
    }

    #[tokio::test]
    async fn missing_usage_is_reported_as_none() {
        let events = ok_events(collect(serve(200, vec![], NO_USAGE).url).await);
        assert_eq!(
            events.last(),
            Some(&LlmEvent::Usage(Usage {
                input_tokens: None,
                output_tokens: None,
                model: Some("gpt-test".into()),
            }))
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, LlmEvent::Usage(_)))
                .count(),
            1
        );
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
    async fn error_body_is_a_non_retryable_stream_error() {
        let items = collect(serve(200, vec![], ERROR_BODY).url).await;
        match items.last() {
            Some(Err(e @ LlmError::Stream { message, .. })) => {
                assert!(!e.is_retryable());
                assert!(message.contains("quota exceeded"));
            }
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
    async fn no_key_sends_no_authorization_header() {
        let served = serve(200, vec![], TEXT);
        let stream = backend(served.url.clone(), None)
            .stream(req())
            .await
            .expect("stream opens");
        let _: Vec<_> = stream.collect().await;
        let headers = served.last_headers.lock().unwrap().clone().unwrap();
        assert!(headers.get("authorization").is_none());
        assert_eq!(
            served.last_path.lock().unwrap().as_deref(),
            Some("/chat/completions")
        );
    }

    #[tokio::test]
    async fn missing_key_names_the_env_var_and_sends_nothing() {
        let served = serve(200, vec![], TEXT);
        let b = OpenAiBackend::new(
            reqwest::Client::new(),
            served.url.clone(),
            None,
            Some("OPENAI_API_KEY".into()),
        );
        let err = b.stream(req()).await.err().unwrap();
        assert_eq!(err, LlmError::Config("OPENAI_API_KEY is not set".into()));
        assert!(served.last_body.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn status_503_is_retryable_and_400_is_not() {
        let served = serve(503, vec![], r#"{"error":{"message":"busy"}}"#);
        let err = backend(served.url, Some("k"))
            .stream(req())
            .await
            .err()
            .unwrap();
        assert!(matches!(err, LlmError::Http { status: 503, .. }));
        assert!(err.is_retryable());

        let served = serve(400, vec![], r#"{"error":{"message":"bad"}}"#);
        let err = backend(served.url, Some("k"))
            .stream(req())
            .await
            .err()
            .unwrap();
        assert_eq!(
            err,
            LlmError::Http {
                status: 400,
                retry_after: None,
                message: "bad".into()
            }
        );
        assert!(!err.is_retryable());
    }
}
