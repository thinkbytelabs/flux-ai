//! End-to-end proof that a tool stage reaches a real MCP server over stdio: the
//! `mcp_fixture` child process, a real `McpTransport` built via `connect_all`,
//! and `run_workflow` driving a workflow whose only stage is a tool stage.
//! Zero LLM calls — `run_workflow` takes an LLM registry by signature, but a
//! tool-only workflow never uses it.
//!
//! This is what substantiates "MCP is a peer transport": the workflow YAML
//! below names a tool and says nothing about transports.

use std::collections::HashMap;

#[tokio::test]
async fn tool_stage_reaches_a_real_mcp_server() {
    let servers = vec![flux_tool_client::mcp::McpServer {
        command: env!("CARGO_BIN_EXE_mcp_fixture").to_string(),
        args: Vec::new(),
    }];
    let transport = flux_tool_client::connect_all(&[], &servers)
        .await
        .expect("spawn the MCP fixture and build the transport");

    let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
        "workflow_id: mcp-tool-stage-smoke\nstages:\n  - id: shout\n    tool: echo_upper\n    args:\n      text: \"{{ workflow.input }}\"\n",
    )
    .expect("parse workflow");

    // No LLM call is made: a tool-only workflow never resolves a provider.
    let llm = flux_llm::LlmRegistry::new();
    let (tx, _rx) = tokio::sync::mpsc::channel(64);

    let outcome = flux_orchestrator::run_workflow(
        &wf,
        &HashMap::new(),
        &llm,
        &transport,
        "hello from flux",
        None,
        None,
        &flux_orchestrator::NoMemory,
        flux_orchestrator::ResumeState::default(),
        &tx,
        flux_orchestrator::RunTuning::default(),
    )
    .await
    .expect("workflow run");

    let flux_orchestrator::RunOutcome::Completed(done) = outcome else {
        panic!("expected the run to complete, not pause");
    };
    let out = done.get("shout").expect("shout stage completed");
    let structured = out.structured.as_ref().expect("tool result is structured");
    assert_eq!(structured["text"], "HELLO FROM FLUX");
    assert_eq!(structured["content"][0]["text"], "HELLO FROM FLUX");
    assert!(structured["structured_content"].is_null());
}

#[tokio::test]
async fn a_tool_name_exported_by_two_mcp_servers_is_a_startup_error() {
    let one = flux_tool_client::mcp::McpServer {
        command: env!("CARGO_BIN_EXE_mcp_fixture").to_string(),
        args: Vec::new(),
    };
    // `.expect_err()` would require `MultiTransport<AnyTransport>: Debug`, but
    // `AnyTransport` deliberately does not derive `Debug` (see its definition in
    // flux-tool-client/src/lib.rs) — so the error is pulled out via `match`
    // instead, which needs no such bound.
    let err = match flux_tool_client::connect_all(&[], &[one.clone(), one]).await {
        Ok(_) => panic!("two servers exporting echo_upper must not both register"),
        Err(e) => e,
    };
    match err {
        flux_core::error::FluxError::Config(m) => {
            assert!(m.contains("echo_upper"), "names the clashing tool: {m}");
        }
        other => panic!("expected Config error, got {other:?}"),
    }
}
