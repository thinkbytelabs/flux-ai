use thiserror::Error;

#[derive(Debug, Error)]
pub enum FluxError {
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("api error {status}: {message}")]
    Api { status: u16, message: String },
}

pub type Result<T> = std::result::Result<T, FluxError>;
