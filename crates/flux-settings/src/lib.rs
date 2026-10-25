#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! Central typed configuration: layered YAML (`base.yaml` -> `<env>.yaml`)
//! with env-var overrides. Loaded once per binary and threaded into
//! constructors. Field defaults equal the constants they replaced.

use flux_core::error::{FluxError, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

impl Settings {
    /// Full load: `FLUX_CONFIG_DIR` (default `./config`) locates the dir,
    /// `FLUX_ENV` (default `dev`) selects the overlay, then env vars override.
    pub fn load() -> Result<Settings> {
        let dir = std::env::var("FLUX_CONFIG_DIR").unwrap_or_else(|_| "./config".to_string());
        let env = std::env::var("FLUX_ENV").unwrap_or_else(|_| "dev".to_string());
        let mut settings = Settings::load_from_dir(Path::new(&dir), &env)?;
        settings.apply_env_overrides();
        settings.config_dir = dir;
        settings.env_name = env;
        Ok(settings)
    }

    /// Apply the explicit, named env-var overrides (highest precedence).
    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("REDIS_URL") {
            self.redis.url = v;
        }
        if let Ok(v) = std::env::var("DATABASE_URL") {
            self.database.url = Some(v);
        }
        if let Ok(v) = std::env::var("FLUX_LOG_FORMAT") {
            self.telemetry.log_format = v;
        }
        if let Ok(v) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            self.telemetry.otlp_endpoint = Some(v);
        }
    }

    /// Load `dir/base.yaml` (required) and deep-merge `dir/<env>.yaml`
    /// (optional) on top, then deserialize. Env-var overrides are applied
    /// separately by [`Settings::load`].
    pub fn load_from_dir(dir: &Path, env: &str) -> Result<Settings> {
        let base_path = dir.join("base.yaml");
        let base_text = std::fs::read_to_string(&base_path)
            .map_err(|e| FluxError::Config(format!("reading {}: {e}", base_path.display())))?;
        let mut base: serde_yaml_ng::Value = serde_yaml_ng::from_str(&base_text)
            .map_err(|e| FluxError::Config(format!("parsing {}: {e}", base_path.display())))?;

        let ovl_path = dir.join(format!("{env}.yaml"));
        let overlay: serde_yaml_ng::Value = match std::fs::read_to_string(&ovl_path) {
            Ok(text) => serde_yaml_ng::from_str(&text)
                .map_err(|e| FluxError::Config(format!("parsing {}: {e}", ovl_path.display())))?,
            // A missing overlay means "no overrides for this env" — only
            // base.yaml is required. A present-but-broken overlay still errors.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_yaml_ng::Value::Null,
            Err(e) => {
                return Err(FluxError::Config(format!(
                    "reading {}: {e}",
                    ovl_path.display()
                )))
            }
        };

        if !overlay.is_null() {
            deep_merge(&mut base, overlay);
        }
        serde_yaml_ng::from_value(base).map_err(|e| FluxError::Config(e.to_string()))
    }
}

/// Recursively merge `overlay` onto `base`: mapping keys are merged key-by-key
/// (overlay wins on scalar/leaf collisions); non-mapping values replace wholesale.
fn deep_merge(base: &mut serde_yaml_ng::Value, overlay: serde_yaml_ng::Value) {
    use serde_yaml_ng::Value::Mapping;
    match (base, overlay) {
        (Mapping(base_map), Mapping(ovl_map)) => {
            for (k, v) in ovl_map {
                match base_map.get_mut(&k) {
                    Some(existing) => deep_merge(existing, v),
                    None => {
                        base_map.insert(k, v);
                    }
                }
            }
        }
        (slot, other) => *slot = other,
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    #[serde(default)]
    pub server: ServerCfg,
    #[serde(default)]
    pub redis: RedisCfg,
    #[serde(default)]
    pub database: DatabaseCfg,
    #[serde(default)]
    pub tools: ToolsCfg,
    #[serde(default)]
    pub llm: LlmCfg,
    #[serde(default)]
    pub orchestrator: OrchestratorCfg,
    #[serde(default)]
    pub sse: SseCfg,
    #[serde(default)]
    pub telemetry: TelemetryCfg,
    #[serde(default)]
    pub agent_defaults: AgentDefaultsCfg,
    /// Where this configuration was loaded from — filled by `load()`, not by
    /// the YAML itself; logged at startup so "which config is active?" has an
    /// answer.
    #[serde(skip)]
    pub config_dir: String,
    #[serde(skip)]
    pub env_name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerCfg {
    #[serde(default = "d_listen")]
    pub listen: String,
}
impl Default for ServerCfg {
    fn default() -> Self {
        ServerCfg { listen: d_listen() }
    }
}
fn d_listen() -> String {
    "0.0.0.0:8080".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedisCfg {
    #[serde(default = "d_redis_url")]
    pub url: String,
    #[serde(default = "d_redis_ttl")]
    pub ttl_secs: u64,
}
impl Default for RedisCfg {
    fn default() -> Self {
        RedisCfg {
            url: d_redis_url(),
            ttl_secs: d_redis_ttl(),
        }
    }
}
fn d_redis_url() -> String {
    "redis://127.0.0.1:6379".to_string()
}
fn d_redis_ttl() -> u64 {
    3600
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseCfg {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "d_pg_max_conns")]
    pub max_connections: u32,
    #[serde(default = "d_pg_acquire")]
    pub acquire_timeout_secs: u64,
}
impl Default for DatabaseCfg {
    fn default() -> Self {
        DatabaseCfg {
            url: None,
            max_connections: d_pg_max_conns(),
            acquire_timeout_secs: d_pg_acquire(),
        }
    }
}
fn d_pg_max_conns() -> u32 {
    10
}
fn d_pg_acquire() -> u64 {
    30
}

/// How to launch one MCP tool server as a child process.
///
/// The child does not inherit Flux's environment: it gets a few non-secret
/// variables (`PATH`, `HOME`, locale, ...) plus the ones named in `env`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerCfg {
    /// Executable to run, e.g. `npx`. Resolved via `PATH`.
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Names of environment variables to pass to this server, e.g.
    /// `[GITHUB_PERSONAL_ACCESS_TOKEN]`. Names only: values stay in the
    /// environment. A listed variable that is unset is skipped.
    #[serde(default)]
    pub env: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolsCfg {
    /// gRPC tool-server address (e.g. `"http://127.0.0.1:50051"`).
    /// Empty string (the default) disables the gRPC tool transport entirely.
    #[serde(default = "d_tool_endpoint")]
    pub endpoint: String,
    /// Additional gRPC tool-server addresses. When non-empty this wins over
    /// `endpoint`, and tools are routed by name across all of them.
    #[serde(default)]
    pub endpoints: Vec<String>,
    /// MCP servers to launch as child processes. Their tools join the same
    /// name-routed table as the gRPC endpoints above.
    #[serde(default)]
    pub mcp: Vec<McpServerCfg>,
}
impl Default for ToolsCfg {
    fn default() -> Self {
        ToolsCfg {
            endpoint: d_tool_endpoint(),
            endpoints: Vec::new(),
            mcp: Vec::new(),
        }
    }
}
fn d_tool_endpoint() -> String {
    String::new()
}

impl ToolsCfg {
    /// Every configured endpoint, plural field winning over the singular one.
    pub fn resolved_endpoints(&self) -> Vec<String> {
        if !self.endpoints.is_empty() {
            return self.endpoints.clone();
        }
        if !self.endpoint.is_empty() {
            return vec![self.endpoint.clone()];
        }
        Vec::new()
    }

    /// Whether any tool backend at all is configured — gRPC or MCP.
    ///
    /// Replaces the older "no gRPC endpoint means no tool support" assumption,
    /// which is wrong now that MCP is a peer transport.
    pub fn has_backends(&self) -> bool {
        !self.resolved_endpoints().is_empty() || !self.mcp.is_empty()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmCfg {
    #[serde(default = "d_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "d_base_delay")]
    pub base_delay_ms: u64,
    /// LLM providers by name, overlaid on the built-in `anthropic` and `openai`.
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderCfg>,
}
impl Default for LlmCfg {
    fn default() -> Self {
        LlmCfg {
            max_attempts: d_max_attempts(),
            base_delay_ms: d_base_delay(),
            providers: BTreeMap::new(),
        }
    }
}

/// One named LLM provider. `api_key_env` names an environment variable; keys
/// never appear in config files.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCfg {
    pub kind: ProviderKind,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    OpenaiCompatible,
}
fn d_max_attempts() -> u32 {
    3
}
fn d_base_delay() -> u64 {
    200
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrchestratorCfg {
    #[serde(default = "d_history_window")]
    pub history_window: usize,
}
impl Default for OrchestratorCfg {
    fn default() -> Self {
        OrchestratorCfg {
            history_window: d_history_window(),
        }
    }
}
fn d_history_window() -> usize {
    20
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SseCfg {
    #[serde(default = "d_tail_poll")]
    pub tail_poll_ms: u64,
}
impl Default for SseCfg {
    fn default() -> Self {
        SseCfg {
            tail_poll_ms: d_tail_poll(),
        }
    }
}
fn d_tail_poll() -> u64 {
    50
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryCfg {
    #[serde(default = "d_log_format")]
    pub log_format: String,
    #[serde(default)]
    pub otlp_endpoint: Option<String>,
}
impl Default for TelemetryCfg {
    fn default() -> Self {
        TelemetryCfg {
            log_format: d_log_format(),
            otlp_endpoint: None,
        }
    }
}
fn d_log_format() -> String {
    "pretty".to_string()
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AgentDefaultsCfg {
    #[serde(default)]
    pub model: ModelDefaultsCfg,
    #[serde(default)]
    pub guardrails: GuardrailDefaultsCfg,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefaultsCfg {
    #[serde(default = "d_profile")]
    pub profile: String,
    #[serde(default = "d_max_tokens")]
    pub max_tokens: u32,
}
impl Default for ModelDefaultsCfg {
    fn default() -> Self {
        ModelDefaultsCfg {
            profile: d_profile(),
            max_tokens: d_max_tokens(),
        }
    }
}
fn d_profile() -> String {
    "claude-opus-4-8".to_string()
}
fn d_max_tokens() -> u32 {
    4096
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailDefaultsCfg {
    #[serde(default = "d_max_iterations")]
    pub max_iterations: u32,
}
impl Default for GuardrailDefaultsCfg {
    fn default() -> Self {
        GuardrailDefaultsCfg {
            max_iterations: d_max_iterations(),
        }
    }
}
fn d_max_iterations() -> u32 {
    4
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn write(dir: &std::path::Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn resolved_endpoints_prefers_plural_then_singular() {
        let mut cfg = ToolsCfg::default();
        assert!(cfg.resolved_endpoints().is_empty());

        cfg.endpoint = "http://a:50051".into();
        assert_eq!(cfg.resolved_endpoints(), vec!["http://a:50051".to_string()]);

        cfg.endpoints = vec!["http://b:50051".into(), "http://c:50051".into()];
        assert_eq!(
            cfg.resolved_endpoints(),
            vec!["http://b:50051".to_string(), "http://c:50051".to_string()]
        );
    }

    #[test]
    fn env_vars_override_file_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("flux-cfg-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "base.yaml", "redis:\n  url: \"redis://file/\"\n");
        write(&dir, "dev.yaml", "{}\n");
        std::env::set_var("REDIS_URL", "redis://env-wins/");
        std::env::set_var("DATABASE_URL", "postgres://env/db");
        std::env::set_var("FLUX_LOG_FORMAT", "json");
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317");

        let mut s = Settings::load_from_dir(&dir, "dev").unwrap();
        s.apply_env_overrides();

        for k in [
            "REDIS_URL",
            "DATABASE_URL",
            "FLUX_LOG_FORMAT",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        ] {
            std::env::remove_var(k);
        }

        assert_eq!(s.redis.url, "redis://env-wins/");
        assert_eq!(s.database.url.as_deref(), Some("postgres://env/db"));
        assert_eq!(s.telemetry.log_format, "json");
        assert_eq!(
            s.telemetry.otlp_endpoint.as_deref(),
            Some("http://otel:4317")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loads_base_and_merges_overlay() {
        let dir = std::env::temp_dir().join(format!("flux-cfg-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(
            &dir,
            "base.yaml",
            "redis:\n  url: \"redis://base/\"\n  ttl_secs: 100\n",
        );
        write(&dir, "prod.yaml", "redis:\n  url: \"redis://prod/\"\n");
        let s = Settings::load_from_dir(&dir, "prod").unwrap();
        // overlay overrides url; base ttl_secs survives the deep-merge
        assert_eq!(s.redis.url, "redis://prod/");
        assert_eq!(s.redis.ttl_secs, 100);
        // unspecified sections fall to struct defaults
        assert_eq!(s.server.listen, "0.0.0.0:8080");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_base_errors() {
        let dir = std::env::temp_dir().join(format!("flux-cfg-nobase-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "dev.yaml", "redis:\n  url: x\n");
        assert!(Settings::load_from_dir(&dir, "dev").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_overlay_is_ok_and_base_applies() {
        let dir = std::env::temp_dir().join(format!("flux-cfg-noovl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "base.yaml", "redis:\n  url: \"redis://only-base/\"\n");
        let s = Settings::load_from_dir(&dir, "staging").unwrap();
        assert_eq!(s.redis.url, "redis://only-base/");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_present_but_broken_overlay_still_errors() {
        let dir = std::env::temp_dir().join(format!("flux-cfg-badovl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "base.yaml", "{}\n");
        write(&dir, "dev.yaml", ":\n  not yaml [\n");
        assert!(Settings::load_from_dir(&dir, "dev").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_field_rejected() {
        let dir = std::env::temp_dir().join(format!("flux-cfg-unknown-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "base.yaml", "redis:\n  bogus_field: 1\n");
        write(&dir, "dev.yaml", "{}\n");
        assert!(Settings::load_from_dir(&dir, "dev").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn defaults_match_old_constants() {
        let s = Settings::default();
        assert_eq!(s.server.listen, "0.0.0.0:8080");
        assert_eq!(s.redis.url, "redis://127.0.0.1:6379");
        assert_eq!(s.redis.ttl_secs, 3600);
        assert_eq!(s.database.url, None);
        assert_eq!(s.database.max_connections, 10);
        assert_eq!(s.database.acquire_timeout_secs, 30);
        assert_eq!(s.tools.endpoint, "");
        assert_eq!(s.llm.max_attempts, 3);
        assert_eq!(s.llm.base_delay_ms, 200);
        assert_eq!(s.orchestrator.history_window, 20);
        assert_eq!(s.sse.tail_poll_ms, 50);
        assert_eq!(s.telemetry.log_format, "pretty");
        assert_eq!(s.telemetry.otlp_endpoint, None);
        assert_eq!(s.agent_defaults.model.profile, "claude-opus-4-8");
        assert_eq!(s.agent_defaults.model.max_tokens, 4096);
        assert_eq!(s.agent_defaults.guardrails.max_iterations, 4);
    }

    #[test]
    fn repo_config_dir_loads_for_all_envs() {
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config"));
        for env in ["dev", "staging", "prod"] {
            let s = Settings::load_from_dir(dir, env).expect("repo config must load");
            // base value present after merge
            assert_eq!(s.server.listen, "0.0.0.0:8080");
            assert_eq!(s.llm.max_attempts, 3);
        }
    }

    #[test]
    fn tools_cfg_defaults_have_no_mcp_servers() {
        let cfg = ToolsCfg::default();
        assert!(cfg.mcp.is_empty());
        assert!(!cfg.has_backends());
    }

    #[test]
    fn providers_default_to_empty_and_parse_kinds() {
        assert!(LlmCfg::default().providers.is_empty());
        let cfg: LlmCfg = serde_yaml_ng::from_str(
            "providers:\n  ollama:\n    kind: openai_compatible\n    base_url: \"http://localhost:11434/v1\"\n",
        )
        .expect("parse llm cfg");
        let ollama = &cfg.providers["ollama"];
        assert_eq!(ollama.kind, ProviderKind::OpenaiCompatible);
        assert_eq!(
            ollama.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
        assert_eq!(ollama.api_key_env, None);
    }

    #[test]
    fn unknown_provider_field_is_rejected() {
        let r: std::result::Result<LlmCfg, _> =
            serde_yaml_ng::from_str("providers:\n  x:\n    kind: anthropic\n    api_key: nope\n");
        assert!(r.is_err());
    }

    #[test]
    fn mcp_servers_parse_from_yaml() {
        let cfg: ToolsCfg = serde_yaml_ng::from_str(
            "mcp:\n  - command: npx\n    args: [\"-y\", \"@modelcontextprotocol/server-github\"]\n  - command: uvx\n",
        )
        .expect("parse tools cfg");
        assert_eq!(cfg.mcp.len(), 2);
        assert_eq!(cfg.mcp[0].command, "npx");
        assert_eq!(
            cfg.mcp[0].args,
            vec!["-y", "@modelcontextprotocol/server-github"]
        );
        assert!(cfg.mcp[1].args.is_empty(), "args defaults to empty");
    }

    #[test]
    fn has_backends_is_true_for_mcp_only_and_grpc_only() {
        let mcp_only: ToolsCfg =
            serde_yaml_ng::from_str("mcp:\n  - command: npx\n").expect("parse mcp-only");
        assert!(mcp_only.has_backends());

        let grpc_only: ToolsCfg = serde_yaml_ng::from_str("endpoint: \"http://127.0.0.1:50051\"\n")
            .expect("parse grpc-only");
        assert!(grpc_only.has_backends());
    }

    #[test]
    fn unknown_mcp_field_is_rejected() {
        let err = serde_yaml_ng::from_str::<ToolsCfg>("mcp:\n  - command: npx\n    cwd: /tmp\n");
        assert!(err.is_err(), "deny_unknown_fields must reject `cwd`");
    }

    #[test]
    fn mcp_env_lists_variable_names() {
        let cfg: ToolsCfg = serde_yaml_ng::from_str(
            "mcp:\n  - command: npx\n    env: [GITHUB_TOKEN, OTHER]\n  - command: uvx\n",
        )
        .expect("parse tools cfg");
        assert_eq!(cfg.mcp[0].env, vec!["GITHUB_TOKEN", "OTHER"]);
        assert!(cfg.mcp[1].env.is_empty(), "env defaults to empty");
    }

    #[test]
    fn mcp_env_values_in_config_are_rejected() {
        // Secrets stay in the environment, never in config.
        let err = serde_yaml_ng::from_str::<ToolsCfg>("mcp:\n  - command: npx\n    env: {A: b}\n");
        assert!(err.is_err(), "a name -> value map must not parse");
    }
}
