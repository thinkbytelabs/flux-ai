#![recursion_limit = "256"]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use anyhow::{bail, Context};
use flux_config::load_registry;
use flux_core::redact::redact_url;
use flux_llm::LlmRegistry;
use flux_memory::{PostgresMemory, RedisMemory, TieredMemory};
use flux_server::{router, AppState, OrchestratorLauncher};
use flux_session::RedisStore;
use flux_workflow::load_workflow;
use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Settings load runs before tracing is initialized, so a
            // tracing::error! here can vanish; stderr always exists.
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let settings = flux_settings::Settings::load().context("loading config")?;
    let _telemetry_guard = flux_core::telemetry::init_tracing(
        false,
        flux_core::telemetry::log_format_from_str(&settings.telemetry.log_format),
    );
    tracing::info!(config_dir = %settings.config_dir, env = %settings.env_name, "configuration loaded");

    // usage: flux-server --agents-dir <dir> --workflows-dir <dir>
    let mut agents_dir: Option<String> = None;
    let mut workflows_dir: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--agents-dir" => agents_dir = Some(args.next().context("--agents-dir needs a path")?),
            "--workflows-dir" => {
                workflows_dir = Some(args.next().context("--workflows-dir needs a path")?)
            }
            other => bail!("unexpected argument: {other}"),
        }
    }
    let agents_dir = agents_dir.context("--agents-dir is required")?;
    let workflows_dir = workflows_dir.context("--workflows-dir is required")?;

    let agent_defaults = flux_config::AgentDefaults {
        profile: settings.agent_defaults.model.profile.clone(),
        max_tokens: settings.agent_defaults.model.max_tokens,
        max_iterations: settings.agent_defaults.guardrails.max_iterations,
    };
    let registry = load_registry(Path::new(&agents_dir), &agent_defaults)
        .with_context(|| format!("loading agents from {agents_dir}"))?;

    // Load every *.yaml/*.yml workflow in the dir, indexed by workflow_id.
    let mut workflows = HashMap::new();
    for entry in std::fs::read_dir(&workflows_dir)
        .with_context(|| format!("reading workflows dir {workflows_dir}"))?
    {
        let path = entry?.path();
        let is_yaml = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e == "yaml" || e == "yml")
            .unwrap_or(false);
        if !is_yaml {
            continue;
        }
        let wf =
            load_workflow(&path).with_context(|| format!("loading workflow {}", path.display()))?;
        flux_orchestrator::validate_agent_refs(&wf, &registry)
            .with_context(|| format!("validating workflow {}", path.display()))?;
        if workflows.insert(wf.workflow_id.clone(), wf).is_some() {
            bail!("duplicate workflow_id in {}", workflows_dir);
        }
    }

    // Every agent's provider must be known: a typo'd or unconfigured provider
    // is a load-time error, not a run-time one.
    let llm = LlmRegistry::from_settings(&settings.llm).context("building LLM providers")?;
    for cfg in registry.values() {
        llm.check(&cfg.model).with_context(|| {
            format!(
                "agent '{}' uses LLM provider '{}'",
                cfg.agent_id, cfg.model.provider
            )
        })?;
    }

    // Warn about provider keys the loaded workflows will need. The server can
    // still boot — mock workflows keep working — but runs against these
    // workflows will fail until the env var is set.
    let mut missing: Vec<String> = Vec::new();
    for wf in workflows.values() {
        for var in flux_orchestrator::missing_provider_keys(wf, &registry, &llm)? {
            let entry = format!("{var} (workflow '{}')", wf.workflow_id);
            if !missing.contains(&entry) {
                missing.push(entry);
            }
        }
    }
    if !missing.is_empty() {
        tracing::warn!(missing = ?missing, "provider API keys not set; runs on these workflows will fail");
    }

    // Preflight: a workflow needs a tool transport when an agent declares
    // tools or the workflow itself has a tool stage. Scanning only the agent
    // registry (as this check used to) misses tool-stage-only workflows,
    // which then boot fine and die at run time with an opaque
    // "NoopTransport: no tool endpoint configured".
    if !settings.tools.has_backends() {
        let needing: Vec<&str> = workflows
            .values()
            .filter(|wf| flux_orchestrator::needs_tool_transport(&registry, wf))
            .map(|wf| wf.workflow_id.as_str())
            .collect();
        if !needing.is_empty() {
            tracing::warn!(workflows = ?needing, "no tool backend is configured (set tools.endpoint, tools.endpoints, or tools.mcp) but these workflows need a tool transport (an agent declares tools, or a stage calls one directly); they will fail until it is configured");
        }
    }

    let store = RedisStore::connect(&settings.redis.url, settings.redis.ttl_secs)
        .await
        .with_context(|| format!("connecting to Redis at {}", redact_url(&settings.redis.url)))?;
    let memory = match settings.database.url.as_deref() {
        Some(db) => {
            let redis_mem = RedisMemory::connect(&settings.redis.url, settings.redis.ttl_secs, 256)
                .await
                .with_context(|| {
                    format!(
                        "connecting conversation-memory redis at {}",
                        redact_url(&settings.redis.url)
                    )
                })?;
            let pg_mem = PostgresMemory::connect(
                db,
                settings.database.max_connections,
                settings.database.acquire_timeout_secs,
            )
            .await
            .with_context(|| {
                format!(
                    "connecting conversation-memory postgres at {}",
                    redact_url(db)
                )
            })?;
            Some(std::sync::Arc::new(TieredMemory::new(redis_mem, pg_mem)))
        }
        None => None,
    };
    let endpoints = settings.tools.resolved_endpoints();
    let servers = mcp_servers(&settings);
    let transport = if endpoints.is_empty() && servers.is_empty() {
        None
    } else {
        Some(Arc::new(
            flux_tool_client::connect_all(&endpoints, &servers)
                .await
                .with_context(|| {
                    format!("connecting to tool backends: gRPC {endpoints:?}, MCP {servers:?}")
                })?,
        ))
    };
    let launcher = OrchestratorLauncher {
        llm: Arc::new(llm),
        registry: Arc::new(registry),
        memory,
        transport,
        tuning: flux_orchestrator::RunTuning {
            history_window: settings.orchestrator.history_window,
            retry: flux_runner::RetryConfig::from(&settings.llm),
        },
    };
    let state = AppState {
        store,
        launcher,
        workflows: Arc::new(workflows),
        tail_poll_ms: settings.sse.tail_poll_ms,
    };

    let listen = &settings.server.listen;
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding {listen}"))?;
    tracing::info!(%listen, redis_url = %redact_url(&settings.redis.url), "flux-server listening");
    axum::serve(listener, router(state))
        .await
        .context("server error")?;
    Ok(())
}

/// `flux-settings` must not depend on `flux-tool-client`, so the two spec types
/// are mapped here. Two fields, one map.
fn mcp_servers(settings: &flux_settings::Settings) -> Vec<flux_tool_client::mcp::McpServer> {
    settings
        .tools
        .mcp
        .iter()
        .map(|s| flux_tool_client::mcp::McpServer {
            command: s.command.clone(),
            args: s.args.clone(),
        })
        .collect()
}
