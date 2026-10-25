//! Maps a model's provider to the backend that serves it.

use crate::anthropic::AnthropicBackend;
use crate::http;
use crate::mock::MockBackend;
use crate::openai::OpenAiBackend;
use crate::types::LlmBackend;
use crate::RetryConfig;
use flux_config::ModelConfig;
use flux_core::error::{FluxError, Result};
use flux_settings::{LlmCfg, ProviderCfg, ProviderKind};
use std::collections::HashMap;
use std::sync::Arc;

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Named backends. The `mock` provider is built in and needs no registration.
pub struct LlmRegistry {
    backends: HashMap<String, Arc<dyn LlmBackend>>,
    /// Provider name -> `api_key_env` variable that was unset or empty.
    missing_keys: HashMap<String, String>,
}

impl LlmRegistry {
    pub fn new() -> Self {
        LlmRegistry {
            backends: HashMap::new(),
            missing_keys: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, backend: Arc<dyn LlmBackend>) {
        self.backends.insert(name.into(), backend);
    }

    /// Builds every backend from the built-ins plus `cfg.providers`. Each
    /// `api_key_env` variable is read once; an empty value counts as unset.
    pub fn from_settings(cfg: &LlmCfg) -> Result<Self> {
        Self::build(cfg, &|name| std::env::var(name).ok())
    }

    fn build(cfg: &LlmCfg, env: &dyn Fn(&str) -> Option<String>) -> Result<Self> {
        if cfg.providers.contains_key("mock") {
            return Err(FluxError::Config(
                "LLM provider name 'mock' is reserved".into(),
            ));
        }
        let providers = effective_providers(cfg);

        let http = http::client()?;
        let mut reg = LlmRegistry::new();
        for (name, p) in providers {
            let key = p
                .api_key_env
                .as_deref()
                .and_then(|var| env(var).filter(|v| !v.is_empty()));
            if key.is_none() {
                if let Some(var) = &p.api_key_env {
                    reg.missing_keys.insert(name.clone(), var.clone());
                }
            }
            let backend: Arc<dyn LlmBackend> = match p.kind {
                ProviderKind::Anthropic => Arc::new(AnthropicBackend::new(
                    http.clone(),
                    p.base_url.unwrap_or_else(|| ANTHROPIC_BASE_URL.into()),
                    key,
                    p.api_key_env,
                )),
                ProviderKind::OpenaiCompatible => {
                    let base_url = p.base_url.ok_or_else(|| {
                        FluxError::Config(format!(
                            "LLM provider '{name}' (openai_compatible) requires base_url"
                        ))
                    })?;
                    Arc::new(OpenAiBackend::new(
                        http.clone(),
                        base_url,
                        key,
                        p.api_key_env,
                    ))
                }
            };
            reg.register(name, backend);
        }
        Ok(reg)
    }

    /// Looks a backend up by provider name.
    pub fn backend(&self, name: &str) -> Option<Arc<dyn LlmBackend>> {
        self.backends.get(name).cloned()
    }

    pub fn backend_for(&self, model: &ModelConfig) -> Result<Arc<dyn LlmBackend>> {
        if model.provider == "mock" {
            return Ok(Arc::new(match &model.mock_response {
                Some(text) => MockBackend::canned(text.clone()),
                None => MockBackend::echo(),
            }));
        }
        let name = model.provider.as_str();
        self.backend(name)
            .ok_or_else(|| FluxError::Config(format!("unknown LLM provider '{name}'")))
    }

    /// `Ok(Some(var))` when the model's provider needs an API key whose
    /// environment variable `var` is unset or empty.
    pub fn check(&self, model: &ModelConfig) -> Result<Option<String>> {
        self.check_provider(&model.provider)
    }

    fn check_provider(&self, name: &str) -> Result<Option<String>> {
        if name == "mock" {
            return Ok(None);
        }
        if !self.backends.contains_key(name) {
            return Err(FluxError::Config(format!("unknown LLM provider '{name}'")));
        }
        Ok(self.missing_keys.get(name).cloned())
    }
}

/// The built-ins merged with `cfg.providers`. A configured entry that takes a
/// built-in's name replaces it but inherits its `api_key_env` unless it sets
/// its own.
fn effective_providers(cfg: &LlmCfg) -> Vec<(String, ProviderCfg)> {
    let mut providers: Vec<(String, ProviderCfg)> = vec![
        (
            "anthropic".into(),
            ProviderCfg {
                kind: ProviderKind::Anthropic,
                base_url: Some(ANTHROPIC_BASE_URL.into()),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
            },
        ),
        (
            "openai".into(),
            ProviderCfg {
                kind: ProviderKind::OpenaiCompatible,
                base_url: Some(OPENAI_BASE_URL.into()),
                api_key_env: Some("OPENAI_API_KEY".into()),
            },
        ),
    ];
    for (name, p) in &cfg.providers {
        let mut p = p.clone();
        if let Some(pos) = providers.iter().position(|(n, _)| n == name) {
            let (_, builtin) = providers.remove(pos);
            if p.api_key_env.is_none() {
                p.api_key_env = builtin.api_key_env;
            }
        }
        if matches!(p.kind, ProviderKind::Anthropic) && p.base_url.is_none() {
            p.base_url = Some(ANTHROPIC_BASE_URL.into());
        }
        providers.push((name.clone(), p));
    }
    providers
}

impl From<&LlmCfg> for RetryConfig {
    fn from(cfg: &LlmCfg) -> Self {
        RetryConfig {
            max_attempts: cfg.max_attempts,
            base_delay_ms: cfg.base_delay_ms,
        }
    }
}

impl Default for LlmRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockTurn;
    use crate::types::{ChatRequest, LlmEvent, OutputFormat};
    use futures_util::StreamExt;

    fn model(provider: &str) -> ModelConfig {
        ModelConfig {
            provider: provider.into(),
            ..ModelConfig::default()
        }
    }

    #[test]
    fn mock_provider_resolves_without_registration() {
        assert!(LlmRegistry::new().backend_for(&model("mock")).is_ok());
    }

    #[test]
    fn unknown_provider_is_a_config_error() {
        match LlmRegistry::new().backend_for(&model("anthropic")) {
            Err(FluxError::Config(m)) => assert!(m.contains("anthropic")),
            other => panic!("expected config error, got ok={}", other.is_ok()),
        }
    }

    #[tokio::test]
    async fn registered_backend_is_returned() {
        let mut reg = LlmRegistry::new();
        reg.register(
            "anthropic",
            Arc::new(MockBackend::scripted(vec![MockTurn::Text(
                "scripted".into(),
            )])),
        );
        let backend = reg.backend_for(&model("anthropic")).unwrap();
        let req = ChatRequest {
            model: "m".into(),
            system: String::new(),
            messages: vec![],
            tools: vec![],
            output: OutputFormat::Text,
            max_tokens: None,
            temperature: None,
            reasoning: None,
        };
        let events: Vec<_> = backend.stream(req).await.unwrap().collect().await;
        assert_eq!(events[0], Ok(LlmEvent::Text("scripted".into())));
    }

    fn provider(kind: ProviderKind, base_url: Option<&str>, env: Option<&str>) -> ProviderCfg {
        ProviderCfg {
            kind,
            base_url: base_url.map(str::to_owned),
            api_key_env: env.map(str::to_owned),
        }
    }

    fn cfg_with(name: &str, p: ProviderCfg) -> LlmCfg {
        let mut cfg = LlmCfg::default();
        cfg.providers.insert(name.into(), p);
        cfg
    }

    #[test]
    fn builtins_are_registered_without_settings() {
        let reg = LlmRegistry::from_settings(&LlmCfg::default()).unwrap();
        assert!(reg.backend("anthropic").is_some());
        assert!(reg.backend("openai").is_some());
        assert!(reg.backend("nope").is_none());
    }

    #[test]
    fn mock_is_a_reserved_provider_name() {
        let cfg = cfg_with("mock", provider(ProviderKind::Anthropic, None, None));
        match LlmRegistry::from_settings(&cfg) {
            Err(FluxError::Config(m)) => assert!(m.contains("mock"), "{m}"),
            other => panic!("expected config error, got ok={}", other.is_ok()),
        }
    }

    #[test]
    fn openai_compatible_requires_base_url() {
        let cfg = cfg_with(
            "ollama",
            provider(ProviderKind::OpenaiCompatible, None, None),
        );
        match LlmRegistry::from_settings(&cfg) {
            Err(FluxError::Config(m)) => assert!(m.contains("base_url"), "{m}"),
            other => panic!("expected config error, got ok={}", other.is_ok()),
        }
    }

    #[test]
    fn check_reports_the_missing_key_env_var() {
        let var = "FLUX_TEST_UNSET_KEY_8c1f";
        let cfg = cfg_with("custom", provider(ProviderKind::Anthropic, None, Some(var)));
        let reg = LlmRegistry::from_settings(&cfg).unwrap();
        let m = model("anthropic");
        // Built-ins: absent from a clean env or not, check never errors.
        assert!(reg.check(&m).is_ok());
        assert_eq!(reg.check_provider("custom").unwrap(), Some(var.to_string()));
        assert!(reg.check_provider("unknown").is_err());
        assert_eq!(reg.check(&model("mock")).unwrap(), None);
    }

    #[test]
    fn missing_and_empty_keys_are_reported_present_keys_are_not() {
        let lookup = |name: &str| match name {
            "ANTHROPIC_API_KEY" => Some("secret".to_string()),
            "OPENAI_API_KEY" => Some(String::new()),
            _ => None,
        };
        let reg = LlmRegistry::build(&LlmCfg::default(), &lookup).unwrap();
        assert_eq!(reg.check_provider("anthropic").unwrap(), None);
        assert_eq!(
            reg.check_provider("openai").unwrap(),
            Some("OPENAI_API_KEY".to_string())
        );
    }

    #[test]
    fn overriding_a_builtin_keeps_its_key_env() {
        let cfg = cfg_with(
            "anthropic",
            provider(ProviderKind::Anthropic, Some("https://gateway/v1"), None),
        );
        let lookup = |_: &str| None;
        let reg = LlmRegistry::build(&cfg, &lookup).unwrap();
        assert_eq!(
            reg.check_provider("anthropic").unwrap(),
            Some("ANTHROPIC_API_KEY".to_string())
        );
        let seen = std::cell::RefCell::new(Vec::new());
        let lookup = |name: &str| {
            seen.borrow_mut().push(name.to_string());
            (name == "ANTHROPIC_API_KEY").then(|| "secret".to_string())
        };
        let reg = LlmRegistry::build(&cfg, &lookup).unwrap();
        assert!(seen.borrow().iter().any(|n| n == "ANTHROPIC_API_KEY"));
        assert_eq!(reg.check_provider("anthropic").unwrap(), None);
    }

    #[test]
    fn a_configured_openai_override_replaces_the_base_url() {
        let cfg = cfg_with(
            "openai",
            provider(
                ProviderKind::OpenaiCompatible,
                Some("https://gateway/v1"),
                None,
            ),
        );
        let resolved = effective_providers(&cfg);
        let (_, p) = resolved.iter().find(|(n, _)| n == "openai").unwrap();
        assert_eq!(p.base_url.as_deref(), Some("https://gateway/v1"));
        assert_eq!(p.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
        assert_eq!(resolved.iter().filter(|(n, _)| n == "openai").count(), 1);
    }

    #[test]
    fn an_explicit_api_key_env_wins_over_the_builtin_one() {
        let cfg = cfg_with(
            "anthropic",
            provider(ProviderKind::Anthropic, None, Some("MY_KEY")),
        );
        let resolved = effective_providers(&cfg);
        let (_, p) = resolved.iter().find(|(n, _)| n == "anthropic").unwrap();
        assert_eq!(p.api_key_env.as_deref(), Some("MY_KEY"));
    }

    #[test]
    fn anthropic_default_base_url_is_used_when_unset() {
        let cfg = cfg_with("claude2", provider(ProviderKind::Anthropic, None, None));
        let resolved = effective_providers(&cfg);
        for name in ["anthropic", "claude2"] {
            let (_, p) = resolved.iter().find(|(n, _)| n == name).unwrap();
            assert_eq!(p.base_url.as_deref(), Some(ANTHROPIC_BASE_URL), "{name}");
        }
    }

    #[test]
    fn an_empty_key_through_a_configured_provider_counts_as_missing() {
        let cfg = cfg_with(
            "custom",
            provider(ProviderKind::Anthropic, None, Some("CUSTOM_KEY")),
        );
        let lookup = |name: &str| (name == "CUSTOM_KEY").then(String::new);
        let reg = LlmRegistry::build(&cfg, &lookup).unwrap();
        assert_eq!(
            reg.check_provider("custom").unwrap(),
            Some("CUSTOM_KEY".to_string())
        );
    }

    #[tokio::test]
    async fn a_configured_ollama_provider_streams_with_no_code_change() {
        let served = crate::testutil::serve(
            200,
            vec![("content-type", "text/event-stream".into())],
            include_str!("fixtures/openai/text.sse"),
        );
        let cfg = cfg_with(
            "ollama",
            provider(ProviderKind::OpenaiCompatible, Some(&served.url), None),
        );
        let reg = LlmRegistry::from_settings(&cfg).unwrap();
        assert_eq!(reg.check_provider("ollama").unwrap(), None);
        let backend = reg.backend("ollama").unwrap();
        let req = ChatRequest {
            model: "llama".into(),
            system: String::new(),
            messages: vec![],
            tools: vec![],
            output: OutputFormat::Text,
            max_tokens: None,
            temperature: None,
            reasoning: None,
        };
        let events: Vec<_> = backend.stream(req).await.unwrap().collect().await;
        assert!(
            events.iter().any(|e| matches!(e, Ok(LlmEvent::Text(_)))),
            "{events:?}"
        );
        let headers = served.last_headers.lock().unwrap().clone().unwrap();
        assert!(headers.get("authorization").is_none());
    }

    #[tokio::test]
    async fn an_agent_with_provider_ollama_streams_from_its_settings_entry() {
        let served = crate::testutil::serve(
            200,
            vec![("content-type", "text/event-stream".into())],
            include_str!("fixtures/openai/text.sse"),
        );
        let cfg = cfg_with(
            "ollama",
            provider(ProviderKind::OpenaiCompatible, Some(&served.url), None),
        );
        let reg = LlmRegistry::from_settings(&cfg).unwrap();
        let m = model("ollama");
        assert_eq!(reg.check(&m).unwrap(), None);
        let backend = reg.backend_for(&m).unwrap();
        let req = ChatRequest {
            model: "llama".into(),
            system: String::new(),
            messages: vec![],
            tools: vec![],
            output: OutputFormat::Text,
            max_tokens: None,
            temperature: None,
            reasoning: None,
        };
        let events: Vec<_> = backend.stream(req).await.unwrap().collect().await;
        assert!(
            events.iter().any(|e| matches!(e, Ok(LlmEvent::Text(_)))),
            "{events:?}"
        );
    }
}
