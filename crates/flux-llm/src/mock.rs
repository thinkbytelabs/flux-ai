//! Deterministic in-process backend: no network, no API key.

use crate::split_text_into_deltas;
use crate::types::{
    ChatRequest, Content, LlmBackend, LlmError, LlmEvent, LlmStream, Message, Role,
};
use futures_util::future::BoxFuture;
use futures_util::stream;
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Mutex;

/// One scripted reply for [`MockBackend::scripted`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MockTurn {
    Text(String),
    /// `(id, name, args)` per call.
    ToolCalls(Vec<(String, String, Value)>),
    Events(Vec<LlmEvent>),
    /// The backend fails to open the stream.
    Fail(LlmError),
    /// The stream opens and then never yields.
    Stall,
}

enum Mode {
    Echo,
    Canned(String),
    Scripted(Mutex<VecDeque<MockTurn>>),
}

/// Mock backend. Scripted mode is a testing aid.
pub struct MockBackend {
    mode: Mode,
    requests: Mutex<Vec<ChatRequest>>,
}

impl MockBackend {
    /// Streams the last user message back.
    pub fn echo() -> Self {
        Self::with(Mode::Echo)
    }

    /// Streams `text` whatever the input.
    pub fn canned(text: String) -> Self {
        Self::with(Mode::Canned(text))
    }

    /// Plays one turn per request, in order.
    pub fn scripted(turns: Vec<MockTurn>) -> Self {
        Self::with(Mode::Scripted(Mutex::new(turns.into())))
    }

    /// Every request received, in order.
    pub fn requests(&self) -> Vec<ChatRequest> {
        lock(&self.requests).clone()
    }

    fn with(mode: Mode) -> Self {
        MockBackend {
            mode,
            requests: Mutex::new(Vec::new()),
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn text_stream(text: &str) -> LlmStream {
    let events: Vec<Result<LlmEvent, LlmError>> = split_text_into_deltas(text)
        .into_iter()
        .map(|d| Ok(LlmEvent::Text(d)))
        .collect();
    stream::iter(events).boxed()
}

/// Text of the last `User` message (its `Text` content joined); empty if none.
fn last_user_message_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| {
            m.content
                .iter()
                .filter_map(|c| match c {
                    Content::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

impl LlmBackend for MockBackend {
    fn stream(&self, req: ChatRequest) -> BoxFuture<'_, Result<LlmStream, LlmError>> {
        let result = match &self.mode {
            Mode::Echo => Ok(text_stream(&last_user_message_text(&req.messages))),
            Mode::Canned(text) => Ok(text_stream(text)),
            Mode::Scripted(turns) => match lock(turns).pop_front() {
                None => Err(LlmError::Config("mock script exhausted".to_string())),
                Some(MockTurn::Text(t)) => Ok(text_stream(&t)),
                Some(MockTurn::ToolCalls(calls)) => {
                    let events: Vec<Result<LlmEvent, LlmError>> = calls
                        .into_iter()
                        .map(|(id, name, args)| Ok(LlmEvent::ToolCall { id, name, args }))
                        .collect();
                    Ok(stream::iter(events).boxed())
                }
                Some(MockTurn::Events(events)) => {
                    Ok(stream::iter(events.into_iter().map(Ok)).boxed())
                }
                Some(MockTurn::Fail(e)) => Err(e),
                Some(MockTurn::Stall) => Ok(stream::pending().boxed()),
            },
        };
        lock(&self.requests).push(req);
        Box::pin(async move { result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OutputFormat;

    fn req(text: &str) -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text(text.into())],
            }],
            tools: vec![],
            output: OutputFormat::Text,
            max_tokens: None,
            temperature: None,
            reasoning: None,
        }
    }

    async fn collect(b: &MockBackend, r: ChatRequest) -> Result<Vec<LlmEvent>, LlmError> {
        let mut s = b.stream(r).await?;
        let mut out = vec![];
        while let Some(ev) = s.next().await {
            out.push(ev?);
        }
        Ok(out)
    }

    fn concat(events: &[LlmEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn echo_streams_the_last_user_message_in_deltas() {
        let events = collect(&MockBackend::echo(), req("hi there"))
            .await
            .unwrap();
        assert!(events.len() >= 2);
        assert_eq!(concat(&events), "hi there");
    }

    #[tokio::test]
    async fn canned_text_ignores_the_input() {
        let b = MockBackend::canned("done".into());
        assert_eq!(concat(&collect(&b, req("x")).await.unwrap()), "done");
    }

    #[tokio::test]
    async fn scripted_turns_play_in_order_and_record_requests() {
        let call = (
            "c1".to_string(),
            "get_weather".to_string(),
            serde_json::json!({"city": "Paris"}),
        );
        let b = MockBackend::scripted(vec![
            MockTurn::ToolCalls(vec![call.clone()]),
            MockTurn::Text("ok".into()),
        ]);
        let first = collect(&b, req("a")).await.unwrap();
        assert_eq!(
            first,
            vec![LlmEvent::ToolCall {
                id: call.0,
                name: call.1,
                args: call.2
            }]
        );
        assert_eq!(concat(&collect(&b, req("b")).await.unwrap()), "ok");
        assert_eq!(b.requests().len(), 2);
        assert_eq!(b.requests()[1], req("b"));
    }

    #[tokio::test]
    async fn exhausted_script_is_an_error() {
        let b = MockBackend::scripted(vec![MockTurn::Text("a".into()), MockTurn::Text("b".into())]);
        collect(&b, req("1")).await.unwrap();
        collect(&b, req("2")).await.unwrap();
        assert!(matches!(
            collect(&b, req("3")).await,
            Err(LlmError::Config(_))
        ));
    }
}
