#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use anyhow::{bail, Context};
use flux_config::{load_agent, AgentDefaults};
use flux_core::event::AgentEvent;
use flux_llm::LlmRegistry;
use flux_memory::{ConversationMemory, PostgresMemory, RedisMemory, TieredMemory, Turn};
use flux_runner::{run_agent, NoopTransport};
use std::path::PathBuf;
use std::process::ExitCode;
use tokio::sync::mpsc;

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
    let settings = flux_settings::Settings::load().context("loading settings")?;
    let _telemetry_guard = flux_core::telemetry::init_tracing(
        true,
        flux_core::telemetry::log_format_from_str(&settings.telemetry.log_format),
    );
    tracing::info!(config_dir = %settings.config_dir, env = %settings.env_name, "configuration loaded");

    // usage: run_agent [--conversation-id ID] <agent.yaml> <message>
    let mut conversation_id: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--conversation-id" => {
                conversation_id = Some(args.next().context("--conversation-id needs an id")?);
            }
            _ => positional.push(arg),
        }
    }
    let (path, message) = match positional.as_slice() {
        [p, m] => (PathBuf::from(p), m.clone()),
        _ => bail!("usage: run_agent [--conversation-id ID] <agent.yaml> <message>"),
    };

    let agent_defaults = AgentDefaults {
        profile: settings.agent_defaults.model.profile.clone(),
        max_tokens: settings.agent_defaults.model.max_tokens,
        max_iterations: settings.agent_defaults.guardrails.max_iterations,
    };

    let cfg = load_agent(&path, &agent_defaults)
        .with_context(|| format!("loading agent {}", path.display()))?;

    let llm = LlmRegistry::from_settings(&settings.llm).context("building LLM providers")?;
    let missing_key = llm.check(&cfg.model).with_context(|| {
        format!(
            "agent '{}' uses LLM provider '{}'",
            cfg.agent_id, cfg.model.provider
        )
    })?;
    if let Some(key_var) = missing_key {
        bail!("{key_var} is not set");
    }

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(64);
    let printer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                AgentEvent::Started { agent_id } => println!("[STARTED] {agent_id}"),
                AgentEvent::Thought { text } => print!("[THOUGHT] {text}"),
                AgentEvent::Text { delta } => print!("{delta}"),
                AgentEvent::ToolCall { tool, args } => println!("\n[TOOL_CALL] {tool} {args}"),
                AgentEvent::ToolResult { tool, result } => {
                    println!("[TOOL_RESULT] {tool} {result}")
                }
                AgentEvent::ResponseComplete { output } => {
                    println!("\n[RESPONSE] {}", output.message);
                    if let Some(structured) = &output.structured {
                        match serde_json::to_string_pretty(structured) {
                            Ok(json) => println!("[STRUCTURED]\n{json}"),
                            Err(_) => println!("[STRUCTURED] {structured}"),
                        }
                    }
                }
                AgentEvent::Usage {
                    input_tokens,
                    output_tokens,
                    ..
                } => eprintln!(
                    "[usage] in={} out={}",
                    count(input_tokens),
                    count(output_tokens)
                ),
                AgentEvent::Failed { error } => eprintln!("[FAILED] {error}"),
            }
        }
    });

    let retry = flux_runner::RetryConfig::from(&settings.llm);

    // Chat mode: load prior history from tiered memory (hot redis + durable postgres).
    let memory = match (&conversation_id, &settings.database.url) {
        (Some(_), Some(db_url)) => {
            let redis = RedisMemory::connect(&settings.redis.url, settings.redis.ttl_secs, 256)
                .await
                .context("connecting redis memory")?;
            let postgres = PostgresMemory::connect(
                db_url,
                settings.database.max_connections,
                settings.database.acquire_timeout_secs,
            )
            .await
            .context("connecting postgres memory")?;
            Some(TieredMemory::new(redis, postgres))
        }
        _ => None,
    };
    let history: Vec<flux_memory::Turn> = match (&memory, &conversation_id) {
        (Some(mem), Some(cid)) => mem
            .recent(cid, settings.orchestrator.history_window)
            .await
            .context("loading conversation history")?,
        _ => Vec::new(),
    };

    let result = if cfg.tools.is_empty() {
        run_agent(
            &llm,
            &NoopTransport,
            &cfg,
            &history,
            &message,
            None,
            &tx,
            retry,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
    } else {
        if !settings.tools.has_backends() {
            bail!("agent declares tools but no tool backend is configured; set tools.endpoint / tools.endpoints for a gRPC tool server, or tools.mcp for an MCP server");
        }
        let transport = flux_tool_client::connect_all(
            &settings.tools.resolved_endpoints(),
            &mcp_servers(&settings),
        )
        .await
        .context("connecting to tool backends")?;
        run_agent(
            &llm,
            &transport,
            &cfg,
            &history,
            &message,
            None,
            &tx,
            retry,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
    };
    drop(tx);
    let _ = printer.await;
    let run = result?;

    // Chat mode: persist this turn durably.
    // NOT transactional: if the user turn appends but the assistant turn then
    // fails, the log is left at an odd length, which makes the next recall
    // drop one turn at the boundary, forever. There is no cheap fix spanning
    // the memory backends, so this at least logs the partial state loudly.
    if let (Some(mem), Some(cid)) = (&memory, &conversation_id) {
        if run.aborted.is_some() {
            // The run gave up before producing a reply. `run.turns` already
            // carries the user's question plus a closing "[run stopped: ...]"
            // note (see flux_runner::aborted_run) — persist that honestly
            // instead of recording a synthetic empty assistant turn.
            mem.append_turns(cid, &run.turns)
                .await
                .context("persisting aborted run's turns")?;
        } else {
            mem.append_turn(cid, &Turn::text(flux_memory::Role::User, message.clone()))
                .await
                .context("persisting user turn")?;
            if let Err(e) = mem
                .append_turn(
                    cid,
                    &Turn::text(flux_memory::Role::Assistant, run.output.message.clone()),
                )
                .await
            {
                tracing::error!(
                    conversation_id = %cid,
                    appended = 1,
                    remaining = 1,
                    error = %e,
                    "partial conversation append: user turn persisted, assistant turn failed, log left at an odd length"
                );
                return Err(e).context("persisting assistant turn");
            }
        }
    }

    // The runner returns Ok for an aborted run so the turns above can still
    // be persisted, but the run itself did not succeed: exit nonzero instead
    // of reporting an empty reply as success.
    if let Some(reason) = run.aborted {
        bail!("agent run aborted: {reason}");
    }
    Ok(())
}

/// A token count for display; `?` when the provider did not report it.
fn count(n: Option<u32>) -> String {
    n.map_or_else(|| "?".to_string(), |n| n.to_string())
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
