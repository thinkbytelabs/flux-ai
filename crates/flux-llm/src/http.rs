//! HTTP helpers shared by the provider backends.

use crate::types::LlmStream;
use crate::types::{LlmError, LlmEvent};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::time::Duration;

/// Connect timeout only: a stream may legitimately run for minutes, so stall
/// detection lives in `stream_chat`, not here.
pub fn client() -> Result<reqwest::Client, LlmError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| LlmError::Config(format!("cannot build HTTP client: {}", e.without_url())))
}

pub fn transport_error(e: reqwest::Error) -> LlmError {
    LlmError::Transport(e.without_url().to_string())
}

/// Turns a non-2xx response into `LlmError::Http`, preferring the provider's
/// `error.message` over the raw body.
pub async fn error_for_status(resp: reqwest::Response) -> Result<reqwest::Response, LlmError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    let body = resp.bytes().await.unwrap_or_default();
    let message = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| {
            let head = &body[..body.len().min(500)];
            String::from_utf8_lossy(head).into_owned()
        });
    Err(LlmError::Http {
        status: status.as_u16(),
        retry_after,
        message,
    })
}

/// Parses assembled tool-call arguments. Empty means "no arguments".
pub fn parse_tool_args(id: &str, name: &str, raw: &str) -> LlmEvent {
    if raw.trim().is_empty() {
        return LlmEvent::ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            args: json!({}),
        };
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(args) => LlmEvent::ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            args,
        },
        Err(e) => LlmEvent::InvalidToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            error: e.to_string(),
        },
    }
}

pub(crate) fn token_count(v: &Value) -> Option<u32> {
    v.as_u64().and_then(|n| u32::try_from(n).ok())
}

/// A provider's SSE payload decoder, driven by [`sse_stream`].
pub(crate) trait SseDecoder {
    /// Handles one SSE event; returns the events it produces.
    fn handle(&mut self, event: &str, data: &str) -> Result<Vec<LlmEvent>, LlmError>;
    /// True once the provider's end-of-stream marker has been seen.
    fn done(&self) -> bool;
}

struct StreamState<D> {
    events: Box<
        dyn futures_util::Stream<Item = Result<eventsource_stream::Event, String>> + Send + Unpin,
    >,
    decoder: D,
    pending: VecDeque<Result<LlmEvent, LlmError>>,
    finished: bool,
}

/// Drives `decoder` over the response's SSE stream. A stream that ends
/// before `done()` is a retryable error; a decode error ends the stream.
pub(crate) fn sse_stream<D: SseDecoder + Send + 'static>(
    resp: reqwest::Response,
    decoder: D,
) -> LlmStream {
    let events = resp
        .bytes_stream()
        .eventsource()
        .map(|r| r.map_err(|e| e.to_string()));
    let state = StreamState {
        events: Box::new(events),
        decoder,
        pending: VecDeque::new(),
        finished: false,
    };
    futures_util::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(item) = st.pending.pop_front() {
                return Some((item, st));
            }
            if st.finished {
                return None;
            }
            match st.events.next().await {
                None => {
                    st.finished = true;
                    if !st.decoder.done() {
                        st.pending.push_back(Err(LlmError::Stream {
                            message: "stream ended early".into(),
                            retryable: true,
                        }));
                    }
                }
                Some(Err(message)) => {
                    st.finished = true;
                    st.pending.push_back(Err(LlmError::Stream {
                        message,
                        retryable: true,
                    }));
                }
                Some(Ok(ev)) => match st.decoder.handle(&ev.event, &ev.data) {
                    Ok(out) => {
                        st.pending.extend(out.into_iter().map(Ok));
                        if st.decoder.done() {
                            st.finished = true;
                        }
                    }
                    Err(e) => {
                        st.finished = true;
                        st.pending.push_back(Err(e));
                    }
                },
            }
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_args_are_an_empty_object() {
        assert_eq!(
            parse_tool_args("1", "t", ""),
            LlmEvent::ToolCall {
                id: "1".into(),
                name: "t".into(),
                args: json!({})
            }
        );
    }

    #[test]
    fn invalid_args_become_invalid_tool_call() {
        assert!(matches!(
            parse_tool_args("1", "t", "{\"x\": "),
            LlmEvent::InvalidToolCall { .. }
        ));
    }
}
