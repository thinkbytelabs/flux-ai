#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use anyhow::{bail, Context};
use flux_config::{load_registry, AgentDefaults};
use flux_core::event::AgentEvent;
use flux_core::workflow_event::WorkflowEvent;
use flux_llm::LlmRegistry;
use flux_memory::{NoMemory, PostgresMemory, RedisMemory, TieredMemory};
use flux_orchestrator::{needs_tool_transport, run_workflow};
use flux_runner::NoopTransport;
use flux_workflow::load_workflow;
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

fn print_event(event: WorkflowEvent) {
    match event {
        WorkflowEvent::StageStarted { stage } => println!("[{stage}][STARTED]"),
        WorkflowEvent::StageSkipped { stage } => println!("[{stage}][SKIPPED]"),
        WorkflowEvent::StageWaitingApproval { stage } => println!("[{stage}][WAITING_APPROVAL]"),
        WorkflowEvent::RefineRound {
            stage,
            round,
            passed,
            feedback,
        } => match (passed, feedback) {
            (false, Some(feedback)) => {
                println!("[{stage}][REFINE] round {round} rejected: {feedback}")
            }
            _ => println!("[{stage}][REFINE] round {round} passed"),
        },
        WorkflowEvent::StageComplete { stage, output } => {
            println!("[{stage}][COMPLETE] {}", output.message);
            if let Some(structured) = &output.structured {
                match serde_json::to_string(structured) {
                    Ok(json) => println!("[{stage}][STRUCTURED] {json}"),
                    Err(_) => println!("[{stage}][STRUCTURED] {structured}"),
                }
            }
        }
        WorkflowEvent::Agent { stage, event } => match event {
            AgentEvent::Started { agent_id } => println!("[{stage}][agent:started] {agent_id}"),
            AgentEvent::Thought { text } => print!("[{stage}][THOUGHT] {text}"),
            AgentEvent::Text { delta } => print!("[{stage}][TEXT] {delta}"),
            AgentEvent::ToolCall { tool, args } => println!("[{stage}][TOOL_CALL] {tool} {args}"),
            AgentEvent::ToolResult { tool, result } => {
                println!("[{stage}][TOOL_RESULT] {tool} {result}")
            }
            AgentEvent::ResponseComplete { output } => {
                println!("[{stage}][RESPONSE] {}", output.message)
            }
            AgentEvent::Usage {
                input_tokens,
                output_tokens,
                ..
            } => eprintln!(
                "[{stage}][usage] in={} out={}",
                count(input_tokens),
                count(output_tokens)
            ),
            AgentEvent::Failed { error } => eprintln!("[{stage}][agent:failed] {error}"),
        },
        WorkflowEvent::WorkflowComplete => println!("[WORKFLOW] complete"),
        WorkflowEvent::WorkflowFailed { stage, error } => {
            eprintln!("[WORKFLOW] failed at {stage}: {error}")
        }
        WorkflowEvent::WorkflowCancelled => println!("[WORKFLOW] cancelled"),
    }
}

/// A token count for display; `?` when the provider did not report it.
fn count(n: Option<u32>) -> String {
    n.map_or_else(|| "?".to_string(), |n| n.to_string())
}

#[allow(clippy::too_many_arguments)]
async fn drive<T, M>(
    wf: &flux_workflow::Workflow,
    registry: &std::collections::HashMap<String, flux_config::AgentConfig>,
    llm: &LlmRegistry,
    transport: &T,
    input: &str,
    conversation_id: Option<&str>,
    context: Option<&serde_json::Value>,
    memory: &M,
    tx: &mpsc::Sender<WorkflowEvent>,
    tuning: flux_orchestrator::RunTuning,
) -> anyhow::Result<flux_orchestrator::RunOutcome>
where
    T: flux_tool_client::ToolTransport + Sync,
    M: flux_memory::ConversationMemory + Sync,
{
    Ok(run_workflow(
        wf,
        registry,
        llm,
        transport,
        input,
        conversation_id,
        context,
        memory,
        flux_orchestrator::ResumeState::default(),
        tx,
        tuning,
    )
    .await?)
}

async fn run() -> anyhow::Result<()> {
    let settings = flux_settings::Settings::load().context("loading settings")?;
    let _telemetry_guard = flux_core::telemetry::init_tracing(
        true,
        flux_core::telemetry::log_format_from_str(&settings.telemetry.log_format),
    );
    tracing::info!(config_dir = %settings.config_dir, env = %settings.env_name, "configuration loaded");

    // usage: run_workflow --agents-dir <dir> [--conversation-id ID] [--context-json JSON] <workflow.yaml> <input>
    let mut agents_dir: Option<String> = None;
    let mut conversation_id: Option<String> = None;
    let mut context: Option<serde_json::Value> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--agents-dir" => agents_dir = Some(args.next().context("--agents-dir needs a path")?),
            "--conversation-id" => {
                conversation_id = Some(args.next().context("--conversation-id needs an id")?)
            }
            "--context-json" => {
                let raw = args.next().context("--context-json needs a JSON object")?;
                context =
                    Some(serde_json::from_str(&raw).context("--context-json must be valid JSON")?);
            }
            _ => positional.push(arg),
        }
    }
    let (wf_path, input) = match positional.as_slice() {
        [w, i] => (PathBuf::from(w), i.clone()),
        _ => bail!(
            "usage: run_workflow --agents-dir <dir> [--conversation-id ID] [--context-json JSON] <workflow.yaml> <input>"
        ),
    };
    let agents_dir = agents_dir.context("--agents-dir is required")?;

    let agent_defaults = AgentDefaults {
        profile: settings.agent_defaults.model.profile.clone(),
        max_tokens: settings.agent_defaults.model.max_tokens,
        max_iterations: settings.agent_defaults.guardrails.max_iterations,
    };

    let registry = load_registry(std::path::Path::new(&agents_dir), &agent_defaults)
        .with_context(|| format!("loading agents from {agents_dir}"))?;
    let wf = load_workflow(&wf_path)
        .with_context(|| format!("loading workflow {}", wf_path.display()))?;

    flux_orchestrator::validate_agent_refs(&wf, &registry)
        .with_context(|| format!("validating workflow {}", wf_path.display()))?;

    // Every agent's provider must be known before anything runs.
    let llm = LlmRegistry::from_settings(&settings.llm).context("building LLM providers")?;
    for cfg in registry.values() {
        llm.check(&cfg.model).with_context(|| {
            format!(
                "agent '{}' uses LLM provider '{}'",
                cfg.agent_id, cfg.model.provider
            )
        })?;
    }

    // Preflight the provider API keys the workflow's agents actually need — a
    // missing env var should fail in milliseconds, not surface mid-run and be
    // retried as if it were transient.
    let missing = flux_orchestrator::missing_provider_keys(&wf, &registry, &llm)?;
    if !missing.is_empty() {
        bail!(
            "missing provider API key(s) for this workflow's agents: {}",
            missing.join(", ")
        );
    }

    let (tx, mut rx) = mpsc::channel::<WorkflowEvent>(64);
    let printer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            print_event(event);
        }
    });

    let tuning = flux_orchestrator::RunTuning {
        history_window: settings.orchestrator.history_window,
        retry: flux_runner::RetryConfig::from(&settings.llm),
    };

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
    let cid = conversation_id.as_deref();

    if !settings.tools.has_backends() && needs_tool_transport(&registry, &wf) {
        bail!(
            "workflow uses tools but no tool backend is configured; \
             set tools.endpoint / tools.endpoints for a gRPC tool server, \
             or tools.mcp for an MCP server"
        );
    }
    let use_tools = settings.tools.has_backends();
    let result = match (use_tools, &memory) {
        (true, Some(mem)) => {
            let transport = flux_tool_client::connect_all(
                &settings.tools.resolved_endpoints(),
                &mcp_servers(&settings),
            )
            .await
            .context("connecting to tool backends")?;
            drive(
                &wf,
                &registry,
                &llm,
                &transport,
                &input,
                cid,
                context.as_ref(),
                mem,
                &tx,
                tuning,
            )
            .await
        }
        (true, None) => {
            let transport = flux_tool_client::connect_all(
                &settings.tools.resolved_endpoints(),
                &mcp_servers(&settings),
            )
            .await
            .context("connecting to tool backends")?;
            drive(
                &wf,
                &registry,
                &llm,
                &transport,
                &input,
                cid,
                context.as_ref(),
                &NoMemory,
                &tx,
                tuning,
            )
            .await
        }
        (false, Some(mem)) => {
            drive(
                &wf,
                &registry,
                &llm,
                &NoopTransport,
                &input,
                cid,
                context.as_ref(),
                mem,
                &tx,
                tuning,
            )
            .await
        }
        (false, None) => {
            drive(
                &wf,
                &registry,
                &llm,
                &NoopTransport,
                &input,
                cid,
                context.as_ref(),
                &NoMemory,
                &tx,
                tuning,
            )
            .await
        }
    };
    drop(tx);
    let _ = printer.await;
    match result? {
        flux_orchestrator::RunOutcome::Completed(_) => {}
        flux_orchestrator::RunOutcome::Paused { waiting } => {
            println!(
                "[WORKFLOW] paused awaiting approval: {}",
                waiting.join(", ")
            );
        }
        // Unreachable: this binary never cancels its run.
        flux_orchestrator::RunOutcome::Cancelled => println!("[WORKFLOW] cancelled"),
    }
    Ok(())
}

/// `flux-settings` must not depend on `flux-tool-client`, so the two spec types
/// are mapped here. Three fields, one map.
fn mcp_servers(settings: &flux_settings::Settings) -> Vec<flux_tool_client::mcp::McpServer> {
    settings
        .tools
        .mcp
        .iter()
        .map(|s| flux_tool_client::mcp::McpServer {
            command: s.command.clone(),
            args: s.args.clone(),
            env: s.env.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_with_tools(tools: &[&str]) -> flux_config::AgentConfig {
        let tools_yaml = tools
            .iter()
            .map(|t| format!("  - {t}"))
            .collect::<Vec<_>>()
            .join("\n");
        let yaml =
            format!("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\ntools:\n{tools_yaml}\n");
        serde_yaml_ng::from_str(&yaml).unwrap()
    }

    fn agent_without_tools() -> flux_config::AgentConfig {
        let yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\n";
        serde_yaml_ng::from_str(yaml).unwrap()
    }

    fn workflow_without_tool_stage() -> flux_workflow::Workflow {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: a\n    input: \"\"\n";
        serde_yaml_ng::from_str(yaml).unwrap()
    }

    fn workflow_with_tool_stage() -> flux_workflow::Workflow {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    tool: t\n";
        serde_yaml_ng::from_str(yaml).unwrap()
    }

    #[test]
    fn no_agent_tools_no_tool_stage_is_false() {
        let mut registry = std::collections::HashMap::new();
        registry.insert("a".to_string(), agent_without_tools());
        assert!(!needs_tool_transport(
            &registry,
            &workflow_without_tool_stage()
        ));
    }

    #[test]
    fn agent_declares_tools_no_tool_stage_is_true() {
        let mut registry = std::collections::HashMap::new();
        registry.insert("a".to_string(), agent_with_tools(&["search"]));
        assert!(needs_tool_transport(
            &registry,
            &workflow_without_tool_stage()
        ));
    }

    #[test]
    fn no_agent_tools_but_a_tool_stage_is_true() {
        // The case the original bug missed: a workflow using only tool
        // STAGES, with no agent declaring tools, still needs a transport.
        let mut registry = std::collections::HashMap::new();
        registry.insert("a".to_string(), agent_without_tools());
        assert!(needs_tool_transport(&registry, &workflow_with_tool_stage()));
    }

    #[test]
    fn agent_tools_and_tool_stage_is_true() {
        let mut registry = std::collections::HashMap::new();
        registry.insert("a".to_string(), agent_with_tools(&["search"]));
        assert!(needs_tool_transport(&registry, &workflow_with_tool_stage()));
    }

    #[test]
    fn a_tool_agent_elsewhere_in_the_registry_does_not_flag_an_unrelated_workflow() {
        // The shipped-samples bug: agents/weather.yaml declares tools, which made
        // `run_workflow workflows/mock-chat.yaml` refuse to start.
        let mut registry = std::collections::HashMap::new();
        registry.insert("a".to_string(), agent_without_tools());
        registry.insert("weather".to_string(), agent_with_tools(&["get_weather"]));
        // workflow_without_tool_stage() references only agent "a"
        assert!(!needs_tool_transport(
            &registry,
            &workflow_without_tool_stage()
        ));
    }
}
