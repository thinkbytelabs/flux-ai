//! Provider-neutral request, event and error types for the Flux LLM layer.

use flux_core::error::FluxError;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use serde_json::Value;
use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub output: OutputFormat,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning: Option<Reasoning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Content {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    ToolResult {
        id: String,
        name: String,
        result: Value,
    },
    /// A provider block the runner echoes back verbatim (e.g. a thinking block).
    Opaque(ProviderBlock),
}

/// A provider's raw content block; `data` is that provider's block JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderBlock {
    pub provider: String,
    pub data: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// `Json` always means "constrain at the provider"; `output_schema_mode:
/// validate` maps to `Text`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OutputFormat {
    Text,
    Json { schema: Value },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reasoning {
    pub effort: Option<String>,
    pub budget_tokens: Option<i32>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LlmEvent {
    Thought(String),
    Text(String),
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    /// A tool call whose arguments did not parse; fed back to the model.
    InvalidToolCall {
        id: String,
        name: String,
        error: String,
    },
    Opaque(ProviderBlock),
    Usage(Usage),
    Done,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LlmError {
    Http {
        status: u16,
        retry_after: Option<Duration>,
        message: String,
    },
    Transport(String),
    Stream {
        message: String,
        retryable: bool,
    },
    Config(String),
    Refused(String),
}

impl LlmError {
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Transport(_) => true,
            LlmError::Http { status, .. } => matches!(status, 408 | 409 | 429 | 500..=599),
            LlmError::Stream { retryable, .. } => *retryable,
            LlmError::Config(_) | LlmError::Refused(_) => false,
        }
    }

    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            LlmError::Http { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Http {
                status, message, ..
            } => write!(f, "HTTP {status}: {message}"),
            LlmError::Transport(m) => write!(f, "transport error: {m}"),
            LlmError::Stream { message, .. } => write!(f, "stream error: {message}"),
            LlmError::Config(m) => write!(f, "{m}"),
            LlmError::Refused(m) => write!(f, "request refused: {m}"),
        }
    }
}

impl std::error::Error for LlmError {}

impl From<LlmError> for FluxError {
    fn from(e: LlmError) -> Self {
        match e {
            LlmError::Config(m) => FluxError::Config(m),
            other => FluxError::Provider(other.to_string()),
        }
    }
}

pub type LlmStream = BoxStream<'static, Result<LlmEvent, LlmError>>;

/// A provider backend. It never emits `Done`: end of stream means done.
pub trait LlmBackend: Send + Sync {
    fn stream(&self, req: ChatRequest) -> BoxFuture<'_, Result<LlmStream, LlmError>>;
}
