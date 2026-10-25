//! End-to-end proof that a tool stage reaches a real gRPC tool server: a real
//! `flux-tool-example` `WeatherTool` service, a real `GrpcTransport` built via
//! `connect_all`, and `run_workflow` driving a workflow whose only stage is a
//! tool stage. Zero LLM calls — `run_workflow` still takes an LLM registry
//! by signature, but a tool-only workflow never uses it.

use std::collections::HashMap;

#[tokio::test]
async fn tool_stage_reaches_a_real_grpc_server() {
    // Bind an OS-assigned ephemeral port first so there's no race between
    // the spawned server binding and the client trying to connect.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(flux_tool_example::serve_on(listener));

    let endpoint = format!("http://{addr}");
    let transport = flux_tool_client::connect_all(&[endpoint], &[])
        .await
        .expect("connect to the example tool server");

    let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
        "workflow_id: tool-stage-smoke\nstages:\n  - id: weather\n    tool: get_weather\n    args:\n      location: \"{{ workflow.input }}\"\n",
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
        "San Francisco",
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
    let out = done.get("weather").expect("weather stage completed");
    let structured = out.structured.as_ref().expect("tool result is structured");
    assert_eq!(structured["location"], "San Francisco");
    assert_eq!(structured["temperature_f"], 68);
    assert_eq!(structured["conditions"], "Partly cloudy");
}
