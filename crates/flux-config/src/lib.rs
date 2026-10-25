#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_core::error::{FluxError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    pub agent_id: String,
    pub display_name: String,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub persona: Option<String>,
    pub system_prompt: String,
    #[serde(default)]
    pub context: Vec<ContextEntry>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub guardrails: Guardrails,
}

fn default_provider() -> String {
    "anthropic".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// The LLM provider's name: a built-in (`anthropic`, `openai`, `mock`) or
    /// an entry under `llm.providers` in settings.
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub thinking_budget_tokens: Option<i32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub mock_response: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            provider: default_provider(),
            profile: None,
            max_tokens: None,
            reasoning_effort: None,
            thinking_budget_tokens: None,
            temperature: None,
            mock_response: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContextEntry {
    Static { text: String },
}

/// How `output_schema` is enforced. `Constrained` (default) also asks the
/// provider to decode against the schema; `Validate` only checks the final
/// text against it, for schemas a provider cannot compile (Anthropic caps
/// union-typed properties and grammar size).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputSchemaMode {
    #[default]
    Constrained,
    Validate,
}

/// Runner guardrails. `max_iterations` bounds the agent's tool-use loop;
/// `output_schema` (when set) validates structured JSON output and, unless
/// `output_schema_mode` is `validate`, forces it at the provider.
/// Later milestones grow this (e.g. token_budget).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Guardrails {
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub output_schema_mode: OutputSchemaMode,
}

impl Guardrails {
    /// The schema to hand the provider for constrained decoding, if any.
    pub fn provider_output_schema(&self) -> Option<&serde_json::Value> {
        match self.output_schema_mode {
            OutputSchemaMode::Constrained => self.output_schema.as_ref(),
            OutputSchemaMode::Validate => None,
        }
    }
}

/// Runtime-sourced fallbacks for agent fields omitted in YAML. Built from
/// `flux_settings::Settings::agent_defaults` by the binary and passed into the
/// loaders.
#[derive(Debug, Clone)]
pub struct AgentDefaults {
    pub profile: String,
    pub max_tokens: u32,
    pub max_iterations: u32,
}

impl AgentConfig {
    /// Fill any unset model/guardrail fields from `defaults`. Idempotent.
    pub fn resolve(&mut self, defaults: &AgentDefaults) {
        if self.model.profile.is_none() {
            self.model.profile = Some(defaults.profile.clone());
        }
        if self.model.max_tokens.is_none() {
            self.model.max_tokens = Some(defaults.max_tokens);
        }
        if self.guardrails.max_iterations.is_none() {
            self.guardrails.max_iterations = Some(defaults.max_iterations);
        }
    }
}

pub fn validate(cfg: &AgentConfig) -> Result<()> {
    if cfg.agent_id.trim().is_empty() {
        return Err(FluxError::Config("agent_id must not be empty".into()));
    }
    if cfg.system_prompt.trim().is_empty() {
        return Err(FluxError::Config("system_prompt must not be empty".into()));
    }
    if let Some(schema) = &cfg.guardrails.output_schema {
        if !schema.is_object() {
            return Err(FluxError::Config(
                "guardrails.output_schema must be a JSON object (a JSON-schema)".into(),
            ));
        }
    }
    Ok(())
}

pub fn load_agent(path: &Path, defaults: &AgentDefaults) -> Result<AgentConfig> {
    let text = std::fs::read_to_string(path)?;
    let mut cfg: AgentConfig =
        serde_yaml_ng::from_str(&text).map_err(|e| FluxError::Config(e.to_string()))?;
    cfg.resolve(defaults);
    validate(&cfg)?;
    Ok(cfg)
}

pub fn load_registry(dir: &Path, defaults: &AgentDefaults) -> Result<HashMap<String, AgentConfig>> {
    let mut registry: HashMap<String, AgentConfig> = HashMap::new();
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let path = entry?.path();
        let is_yaml = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e == "yaml" || e == "yml")
            .unwrap_or(false);
        if !is_yaml {
            continue;
        }
        let cfg = load_agent(&path, defaults)?;
        if let Some(existing) = registry.insert(cfg.agent_id.clone(), cfg) {
            return Err(FluxError::Config(format!(
                "duplicate agent_id '{}' in {}",
                existing.agent_id,
                dir.display()
            )));
        }
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl AgentDefaults {
        fn test() -> Self {
            AgentDefaults {
                profile: "claude-opus-4-8".into(),
                max_tokens: 4096,
                max_iterations: 4,
            }
        }
    }

    const GOOD: &str = r#"
agent_id: echo
display_name: Echo Agent
model:
  profile: claude-opus-4-8
  max_tokens: 1024
persona: "You are terse."
system_prompt: "Repeat the user's message back, rephrased in five words."
context:
  - type: static
    text: "This is a demo agent."
guardrails: {}
"#;

    #[test]
    fn parses_a_valid_agent() {
        let cfg: AgentConfig = serde_yaml_ng::from_str(GOOD).unwrap();
        assert_eq!(cfg.agent_id, "echo");
        assert_eq!(cfg.model.profile.as_deref(), Some("claude-opus-4-8"));
        assert_eq!(cfg.model.max_tokens, Some(1024));
        assert_eq!(cfg.context.len(), 1);
    }

    #[test]
    fn model_fields_default() {
        let cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n")
                .unwrap();
        assert_eq!(cfg.model.profile, None);
        assert_eq!(cfg.model.max_tokens, None);
        assert_eq!(cfg.model.provider, "anthropic");
        assert!(cfg.context.is_empty());
    }

    #[test]
    fn provider_defaults_to_anthropic_and_accepts_any_name() {
        let omitted: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\n").unwrap();
        assert_eq!(omitted.model.provider, "anthropic");
        assert_eq!(ModelConfig::default().provider, "anthropic");
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: ollama\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(cfg.model.provider, "ollama");
    }

    #[test]
    fn parses_openai_provider() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: openai\n  profile: gpt-4o\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(cfg.model.provider, "openai");
        assert_eq!(cfg.model.profile.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn rejects_empty_system_prompt() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: \"\"\nmodel: {}\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&cfg) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn parses_tools_and_max_iterations() {
        let yaml = "agent_id: w\ndisplay_name: W\nsystem_prompt: hi\nmodel: {}\ntools:\n  - get_weather\nguardrails:\n  max_iterations: 6\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(cfg.tools, vec!["get_weather".to_string()]);
        assert_eq!(cfg.guardrails.max_iterations, Some(6));
    }

    #[test]
    fn max_iterations_defaults_to_none_before_resolve() {
        let cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n")
                .unwrap();
        assert!(cfg.tools.is_empty());
        assert_eq!(cfg.guardrails.max_iterations, None);
    }

    #[test]
    fn parses_output_schema() {
        let yaml = "agent_id: c\ndisplay_name: C\nsystem_prompt: hi\nmodel: {}\nguardrails:\n  output_schema:\n    type: object\n    properties:\n      sentiment: { type: string }\n    required: [sentiment]\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let schema = cfg.guardrails.output_schema.expect("schema present");
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"][0], "sentiment");
    }

    #[test]
    fn output_schema_is_sent_to_the_provider_by_default() {
        let yaml = "agent_id: c\ndisplay_name: C\nsystem_prompt: hi\nmodel: {}\nguardrails:\n  output_schema: { type: object }\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(
            cfg.guardrails.output_schema_mode,
            OutputSchemaMode::Constrained
        );
        assert!(cfg.guardrails.provider_output_schema().is_some());
    }

    #[test]
    fn validate_mode_keeps_the_schema_but_not_for_the_provider() {
        let yaml = "agent_id: c\ndisplay_name: C\nsystem_prompt: hi\nmodel: {}\nguardrails:\n  output_schema: { type: object }\n  output_schema_mode: validate\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(
            cfg.guardrails.output_schema_mode,
            OutputSchemaMode::Validate
        );
        assert!(cfg.guardrails.output_schema.is_some());
        assert!(cfg.guardrails.provider_output_schema().is_none());
    }

    #[test]
    fn output_schema_absent_by_default() {
        let cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n")
                .unwrap();
        assert!(cfg.guardrails.output_schema.is_none());
    }

    #[test]
    fn rejects_non_object_output_schema() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\nguardrails:\n  output_schema: \"not-a-schema\"\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&cfg) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn load_registry_indexes_by_agent_id() {
        let dir = std::env::temp_dir().join(format!("flux-reg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.yaml"),
            "agent_id: alpha\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("b.yaml"),
            "agent_id: beta\ndisplay_name: B\nsystem_prompt: hi\nmodel: {}\n",
        )
        .unwrap();
        let reg = load_registry(&dir, &AgentDefaults::test()).unwrap();
        assert_eq!(reg.len(), 2);
        assert_eq!(reg["alpha"].display_name, "A");
        assert!(reg.contains_key("beta"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_registry_rejects_duplicate_agent_id() {
        let dir = std::env::temp_dir().join(format!("flux-reg-dup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.yaml"),
            "agent_id: same\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("b.yaml"),
            "agent_id: same\ndisplay_name: B\nsystem_prompt: hi\nmodel: {}\n",
        )
        .unwrap();
        let err = load_registry(&dir, &AgentDefaults::test());
        std::fs::remove_dir_all(&dir).ok();
        match err {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn load_registry_accepts_repo_agents_dir() {
        // The repo's agents/ dir must load without duplicate-id collisions
        // (the OpenAI variants must carry distinct agent_ids).
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../agents"));
        let reg = load_registry(dir, &AgentDefaults::test()).unwrap();
        // distinct ids present for both providers
        assert!(reg.contains_key("echo"));
        assert!(reg.contains_key("echo-openai"));
        assert!(reg.contains_key("weather"));
        assert!(reg.contains_key("weather-openai"));
    }

    #[test]
    fn parses_reasoning_fields() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: anthropic\n  thinking_budget_tokens: 2048\n  reasoning_effort: high\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(cfg.model.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(cfg.model.thinking_budget_tokens, Some(2048));
    }

    #[test]
    fn parses_mock_provider_and_response() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: mock\n  mock_response: \"canned\"\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(cfg.model.provider, "mock");
        assert_eq!(cfg.model.mock_response.as_deref(), Some("canned"));
    }

    #[test]
    fn mock_response_defaults_none() {
        let cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n")
                .unwrap();
        assert_eq!(cfg.model.mock_response, None);
        assert_eq!(ModelConfig::default().mock_response, None);
    }

    #[test]
    fn reasoning_fields_default_none() {
        let cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n")
                .unwrap();
        assert_eq!(cfg.model.reasoning_effort, None);
        assert_eq!(cfg.model.thinking_budget_tokens, None);
        assert_eq!(ModelConfig::default().reasoning_effort, None);
        assert_eq!(ModelConfig::default().thinking_budget_tokens, None);
    }

    #[test]
    fn resolve_fills_missing_from_defaults() {
        let defaults = AgentDefaults {
            profile: "default-model".into(),
            max_tokens: 1234,
            max_iterations: 9,
        };
        // model omits profile/max_tokens; guardrails omit max_iterations
        let mut cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel: {}\n")
                .unwrap();
        cfg.resolve(&defaults);
        assert_eq!(cfg.model.profile.as_deref(), Some("default-model"));
        assert_eq!(cfg.model.max_tokens, Some(1234));
        assert_eq!(cfg.guardrails.max_iterations, Some(9));
    }

    #[test]
    fn resolve_keeps_explicit_values() {
        let defaults = AgentDefaults {
            profile: "default-model".into(),
            max_tokens: 1234,
            max_iterations: 9,
        };
        let mut cfg: AgentConfig = serde_yaml_ng::from_str(
            "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  profile: explicit\n  max_tokens: 50\nguardrails:\n  max_iterations: 2\n",
        )
        .unwrap();
        cfg.resolve(&defaults);
        assert_eq!(cfg.model.profile.as_deref(), Some("explicit"));
        assert_eq!(cfg.model.max_tokens, Some(50));
        assert_eq!(cfg.guardrails.max_iterations, Some(2));
    }

    #[test]
    fn rejects_an_unknown_agent_field() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nguardrailz: {}\n";
        assert!(serde_yaml_ng::from_str::<AgentConfig>(yaml).is_err());
    }

    #[test]
    fn rejects_an_unknown_model_field() {
        // `max_token` (typo of max_tokens) used to silently fall back to defaults.
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  max_token: 99\n";
        assert!(serde_yaml_ng::from_str::<AgentConfig>(yaml).is_err());
    }

    #[test]
    fn rejects_an_unknown_field_inside_a_context_entry() {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\ncontext:\n  - type: static\n    text: \"x\"\n    extra: 1\n";
        assert!(serde_yaml_ng::from_str::<AgentConfig>(yaml).is_err());
    }
}
