#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_config::{AgentConfig, ContextEntry};
use flux_core::error::{FluxError, Result};
use futures_util::StreamExt;

pub mod anthropic;
pub mod http;
pub mod mock;
pub mod openai;
pub mod registry;
pub mod types;

#[cfg(test)]
mod testutil;

pub use mock::{MockBackend, MockTurn};
pub use registry::LlmRegistry;
pub use types::{
    ChatRequest, Content, LlmBackend, LlmError, LlmEvent, LlmStream, Message, OutputFormat,
    ProviderBlock, Reasoning, Role, ToolDef, Usage,
};

/// Provider retry tuning. `Default` matches the M5f constants (3 attempts,
/// 200ms base) so behavior is unchanged when callers don't override.
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_attempts: 3,
            base_delay_ms: 200,
        }
    }
}

/// Split text into streamed deltas (whitespace-preserving chunks) so the mock
/// exercises the same per-chunk Text path as a real provider. Chunks
/// concatenate back to the original; empty input yields no deltas.
pub fn split_text_into_deltas(text: &str) -> Vec<String> {
    text.split_inclusive(' ').map(|s| s.to_string()).collect()
}

pub fn build_system_prompt(cfg: &AgentConfig) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(persona) = &cfg.persona {
        parts.push(persona.as_str());
    }
    parts.push(cfg.system_prompt.as_str());
    for entry in &cfg.context {
        let ContextEntry::Static { text } = entry;
        parts.push(text.as_str());
    }
    parts.join("\n\n")
}

/// How long a provider may stay silent — before the response starts, or between
/// two stream chunks — before the attempt fails. A backstop against a stalled
/// connection, not a latency target: an idle budget, unlike a total one, never
/// cuts off a long generation that is still streaming.
const LLM_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Wait for the next stream item, or fail once the stream has been silent for `idle`.
async fn next_chunk<S: futures_util::Stream + Unpin>(
    stream: &mut S,
    idle: std::time::Duration,
) -> Option<Result<S::Item>> {
    match tokio::time::timeout(idle, stream.next()).await {
        Ok(item) => item.map(Ok),
        Err(_) => Some(Err(FluxError::Provider(format!(
            "provider stream stalled: no data for {}s",
            idle.as_secs()
        )))),
    }
}

/// Longest `Retry-After` the retry loop will wait for.
const MAX_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(60);

/// Full jitter: a uniformly random delay in `[0, base_delay_ms * 2^(attempt-1)]`.
fn jitter_delay(attempt: u32, base_delay_ms: u64) -> std::time::Duration {
    use std::hash::BuildHasher;
    let bound = base_delay_ms.saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1)));
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let rand = std::collections::hash_map::RandomState::new().hash_one(nanos);
    std::time::Duration::from_millis(rand % bound.saturating_add(1))
}

/// Outcome of one `stream_chat` attempt.
enum ChatAttempt {
    /// Finished cleanly, or the receiver went away.
    Finished,
    Failed {
        error: LlmError,
        emitted: bool,
    },
}

async fn run_chat_attempt(
    backend: &dyn LlmBackend,
    req: &ChatRequest,
    tx: &tokio::sync::mpsc::Sender<LlmEvent>,
    idle: std::time::Duration,
) -> ChatAttempt {
    use LlmEvent as Ev;
    let stalled = |what: &str| LlmError::Stream {
        message: format!("{what} stalled: no data for {}s", idle.as_secs()),
        retryable: true,
    };
    let mut stream = match tokio::time::timeout(idle, backend.stream(req.clone())).await {
        Ok(Ok(s)) => s,
        Ok(Err(error)) => {
            return ChatAttempt::Failed {
                error,
                emitted: false,
            }
        }
        Err(_) => {
            return ChatAttempt::Failed {
                error: stalled("provider response"),
                emitted: false,
            }
        }
    };
    // Everything but Text/Thought is held until the attempt ends cleanly, so a
    // retry can never hand the caller the same tool call twice.
    let mut held: Vec<Ev> = Vec::new();
    let mut emitted = false;
    loop {
        match next_chunk(&mut stream, idle).await {
            None => break,
            Some(Err(_)) => {
                return ChatAttempt::Failed {
                    error: stalled("provider stream"),
                    emitted,
                }
            }
            Some(Ok(Err(error))) => return ChatAttempt::Failed { error, emitted },
            Some(Ok(Ok(ev @ (Ev::Text(_) | Ev::Thought(_))))) => {
                emitted = true;
                if tx.send(ev).await.is_err() {
                    return ChatAttempt::Finished;
                }
            }
            Some(Ok(Ok(Ev::Done))) => {}
            Some(Ok(Ok(ev))) => held.push(ev),
        }
    }
    if !held.iter().any(|e| matches!(e, Ev::Usage(_))) {
        held.push(Ev::Usage(Usage::default()));
    }
    for ev in held {
        if let Ev::Usage(u) = &ev {
            tracing::info!(
                input_tokens = ?u.input_tokens,
                output_tokens = ?u.output_tokens,
                "llm call complete"
            );
            flux_core::metrics::record_llm_call(u.input_tokens, u.output_tokens);
        }
        if tx.send(ev).await.is_err() {
            return ChatAttempt::Finished;
        }
    }
    let _ = tx.send(Ev::Done).await;
    ChatAttempt::Finished
}

/// Stream one chat turn from `backend` into `tx`, ending with `Done`. Retries
/// a retryable failure only while nothing the caller could have seen (Text or
/// Thought) was sent.
pub async fn stream_chat(
    backend: &dyn LlmBackend,
    req: ChatRequest,
    tx: &tokio::sync::mpsc::Sender<LlmEvent>,
    retry: RetryConfig,
) -> Result<()> {
    stream_chat_with_idle(backend, req, tx, retry, LLM_IDLE_TIMEOUT).await
}

pub(crate) async fn stream_chat_with_idle(
    backend: &dyn LlmBackend,
    req: ChatRequest,
    tx: &tokio::sync::mpsc::Sender<LlmEvent>,
    retry: RetryConfig,
    idle: std::time::Duration,
) -> Result<()> {
    let mut attempt: u32 = 1;
    loop {
        match run_chat_attempt(backend, &req, tx, idle).await {
            ChatAttempt::Finished => return Ok(()),
            ChatAttempt::Failed { error, emitted } => {
                if !emitted && attempt < retry.max_attempts && error.is_retryable() {
                    tracing::warn!(attempt, %error, "provider attempt failed; retrying");
                    let delay = error
                        .retry_after()
                        .map(|d| d.min(MAX_RETRY_AFTER))
                        .unwrap_or_else(|| jitter_delay(attempt, retry.base_delay_ms));
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                } else {
                    return Err(error.into());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_stalled_stream_is_an_error_not_a_hang() {
        let mut stalled = futures_util::stream::pending::<u8>();
        let next = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            next_chunk(&mut stalled, std::time::Duration::from_millis(50)),
        )
        .await
        .expect("next_chunk must give up after its idle budget");
        assert!(matches!(next, Some(Err(FluxError::Provider(_)))));
    }

    #[tokio::test]
    async fn next_chunk_passes_items_and_end_of_stream_through() {
        let mut s = futures_util::stream::iter([7u8]);
        let idle = std::time::Duration::from_secs(1);
        assert!(matches!(next_chunk(&mut s, idle).await, Some(Ok(7))));
        assert!(next_chunk(&mut s, idle).await.is_none());
    }

    #[test]
    fn retry_config_default_matches_old_constants() {
        let r = RetryConfig::default();
        assert_eq!(r.max_attempts, 3);
        assert_eq!(r.base_delay_ms, 200);
    }

    #[test]
    fn split_deltas_rejoin_to_original() {
        let parts = split_text_into_deltas("hello mock world");
        assert!(
            parts.len() >= 2,
            "multi-word text should split into multiple chunks"
        );
        assert_eq!(parts.concat(), "hello mock world");
    }

    #[test]
    fn split_deltas_empty_is_empty() {
        assert!(split_text_into_deltas("").is_empty());
    }

    mod chat {
        use super::*;
        use crate::mock::{MockBackend, MockTurn};
        use crate::types::LlmEvent as Ev;
        use futures_util::future::BoxFuture;
        use futures_util::stream;
        use std::collections::VecDeque;
        use std::sync::Mutex;
        use std::time::Duration;

        fn req() -> ChatRequest {
            ChatRequest {
                model: "m".into(),
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

        async fn run(backend: &dyn LlmBackend, retry: RetryConfig) -> (Result<()>, Vec<Ev>) {
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);
            let res = stream_chat(backend, req(), &tx, retry).await;
            drop(tx);
            let mut out = vec![];
            while let Some(e) = rx.recv().await {
                out.push(e);
            }
            (res, out)
        }

        /// Plays one prepared stream per attempt.
        struct Attempts(
            Mutex<VecDeque<Vec<std::result::Result<Ev, LlmError>>>>,
            Mutex<u32>,
        );
        impl Attempts {
            fn new(a: Vec<Vec<std::result::Result<Ev, LlmError>>>) -> Self {
                Attempts(Mutex::new(a.into()), Mutex::new(0))
            }
            fn calls(&self) -> u32 {
                *self.1.lock().unwrap()
            }
        }
        impl LlmBackend for Attempts {
            fn stream(
                &self,
                _req: ChatRequest,
            ) -> BoxFuture<'_, std::result::Result<LlmStream, LlmError>> {
                *self.1.lock().unwrap() += 1;
                let items = self.0.lock().unwrap().pop_front().unwrap_or_default();
                Box::pin(async move { Ok(Box::pin(stream::iter(items)) as LlmStream) })
            }
        }

        fn flaky() -> LlmError {
            LlmError::Stream {
                message: "drop".into(),
                retryable: true,
            }
        }

        fn call() -> Ev {
            Ev::ToolCall {
                id: "c1".into(),
                name: "t".into(),
                args: serde_json::json!({}),
            }
        }

        #[tokio::test]
        async fn stream_chat_forwards_text_then_done() {
            let b = MockBackend::canned("a b".into());
            let (res, out) = run(&b, RetryConfig::default()).await;
            res.unwrap();
            assert_eq!(
                out,
                vec![
                    Ev::Text("a ".into()),
                    Ev::Text("b".into()),
                    Ev::Usage(Usage::default()),
                    Ev::Done
                ]
            );
        }

        #[tokio::test]
        async fn tool_calls_usage_and_opaque_are_held_until_the_attempt_ends() {
            let usage = Usage {
                input_tokens: Some(3),
                output_tokens: Some(4),
                model: None,
            };
            let b = MockBackend::scripted(vec![MockTurn::Events(vec![
                call(),
                Ev::Usage(usage.clone()),
                Ev::Text("x".into()),
            ])]);
            let (res, out) = run(&b, RetryConfig::default()).await;
            res.unwrap();
            assert_eq!(
                out,
                vec![Ev::Text("x".into()), call(), Ev::Usage(usage), Ev::Done]
            );
        }

        #[tokio::test]
        async fn retryable_failure_before_output_is_retried_without_duplicates() {
            let b = Attempts::new(vec![vec![Ok(call()), Err(flaky())], vec![Ok(call())]]);
            let retry = RetryConfig {
                max_attempts: 3,
                base_delay_ms: 1,
            };
            let (res, out) = run(&b, retry).await;
            res.unwrap();
            assert_eq!(out.iter().filter(|e| **e == call()).count(), 1);
            assert_eq!(b.calls(), 2);
        }

        #[tokio::test]
        async fn non_retryable_http_error_is_not_retried() {
            let b = MockBackend::scripted(vec![
                MockTurn::Fail(LlmError::Http {
                    status: 401,
                    retry_after: None,
                    message: "no".into(),
                }),
                MockTurn::Text("never".into()),
            ]);
            let (res, out) = run(&b, RetryConfig::default()).await;
            assert!(res.is_err());
            assert!(out.is_empty());
            assert_eq!(b.requests().len(), 1);
        }

        #[tokio::test]
        async fn failure_after_text_is_not_retried() {
            let b = Attempts::new(vec![
                vec![Ok(Ev::Text("partial".into())), Err(flaky())],
                vec![Ok(Ev::Text("again".into()))],
            ]);
            let retry = RetryConfig {
                max_attempts: 3,
                base_delay_ms: 1,
            };
            let (res, out) = run(&b, retry).await;
            assert!(res.is_err());
            assert_eq!(out, vec![Ev::Text("partial".into())]);
            assert_eq!(b.calls(), 1);
        }

        #[tokio::test(start_paused = true)]
        async fn retry_after_is_honoured_and_capped() {
            for (asked, waited) in [(5, 5), (600, 60)] {
                let b = MockBackend::scripted(vec![
                    MockTurn::Fail(LlmError::Http {
                        status: 429,
                        retry_after: Some(Duration::from_secs(asked)),
                        message: "slow down".into(),
                    }),
                    MockTurn::Text("ok".into()),
                ]);
                let start = tokio::time::Instant::now();
                let (res, _) = run(&b, RetryConfig::default()).await;
                res.unwrap();
                assert_eq!(start.elapsed(), Duration::from_secs(waited));
            }
        }

        #[test]
        fn full_jitter_stays_within_the_exponential_bound() {
            for _ in 0..1000 {
                assert!(jitter_delay(3, 200) <= Duration::from_millis(800));
            }
        }

        #[tokio::test]
        async fn a_stalled_attempt_times_out_and_is_retried() {
            let b = MockBackend::scripted(vec![MockTurn::Stall, MockTurn::Text("ok".into())]);
            let (tx, _rx) = tokio::sync::mpsc::channel(64);
            let retry = RetryConfig {
                max_attempts: 3,
                base_delay_ms: 1,
            };
            stream_chat_with_idle(&b, req(), &tx, retry, Duration::from_millis(50))
                .await
                .unwrap();
            assert_eq!(b.requests().len(), 2);
        }

        #[test]
        fn llm_error_retryability() {
            let http = |status| LlmError::Http {
                status,
                retry_after: None,
                message: String::new(),
            };
            assert!(LlmError::Transport("x".into()).is_retryable());
            for s in [408, 409, 429, 500, 503] {
                assert!(http(s).is_retryable(), "{s}");
            }
            for s in [400, 401, 403, 404, 422] {
                assert!(!http(s).is_retryable(), "{s}");
            }
            assert!(!LlmError::Config("x".into()).is_retryable());
            assert!(!LlmError::Refused("x".into()).is_retryable());
        }
    }
}
