//! `McpTransport` driven against the real `mcp_fixture` child process: manifest
//! discovery in this file's first test, then calls and lifecycle in Tasks 3 and 6.

use flux_tool_client::mcp::{McpServer, McpTransport};
use flux_tool_client::ToolTransport;

fn fixture() -> McpServer {
    McpServer {
        command: env!("CARGO_BIN_EXE_mcp_fixture").to_string(),
        args: Vec::new(),
        env: Vec::new(),
    }
}

#[tokio::test]
async fn list_tools_returns_the_fixtures_manifest() {
    let transport = McpTransport::spawn(&fixture())
        .await
        .expect("spawn and handshake with the MCP fixture");

    let manifest = transport.list_tools().await.expect("list tools");
    let mut names: Vec<&str> = manifest.tools.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(
        names,
        vec!["always_fails", "echo_upper", "emit_structured", "read_env"]
    );

    let echo = manifest.find("echo_upper").expect("echo_upper in manifest");
    assert_eq!(echo.description, "Uppercase the given text");
    assert_eq!(echo.parameters["type"], "object");
    assert_eq!(echo.parameters["required"][0], "text");
}

/// Proves the boot path is bounded, against a real child that completes the
/// handshake and then answers nothing.
///
/// Ignored because it must burn the real `LIST_TOOLS_TIMEOUT` — the budget is a
/// constant, and making it injectable purely for this would widen
/// `McpTransport`'s public surface. The everyday guard is the `within` unit
/// tests in flux-tool-client; this one exists because a handshake-only timeout
/// already shipped once, and only an end-to-end hang proves `tools/list` is
/// actually wired through the budget rather than merely having one defined.
///
/// Run with: `cargo test -p flux-cli --test mcp_transport -- --ignored`
#[tokio::test]
#[ignore = "burns the real 30s tools/list budget (cargo test -- --ignored)"]
async fn a_server_silent_after_the_handshake_cannot_hang_boot() {
    let server = McpServer {
        command: env!("CARGO_BIN_EXE_mcp_fixture").to_string(),
        args: vec!["--silent-after-initialize".to_string()],
        env: Vec::new(),
    };
    // The handshake itself must still succeed — otherwise this would be
    // re-testing HANDSHAKE_TIMEOUT rather than the gap after it.
    let transport = McpTransport::spawn(&server)
        .await
        .expect("handshake succeeds; the silence starts afterwards");

    let started = std::time::Instant::now();
    let err = match transport.list_tools().await {
        Ok(m) => panic!("a silent server must not yield a manifest: {m:?}"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("tools/list"), "names the request: {msg}");
    assert!(msg.contains("within"), "reads as a timeout: {msg}");
    // Bounded, not merely slow.
    assert!(
        started.elapsed() < std::time::Duration::from_secs(90),
        "must give up on its own budget, took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn spawn_reports_a_child_pid() {
    let transport = McpTransport::spawn(&fixture()).await.expect("spawn");
    assert!(
        transport.child_pid().is_some(),
        "PID is needed by the lifecycle test"
    );
}

#[tokio::test]
async fn spawning_a_missing_command_fails_with_a_useful_message() {
    let err = McpTransport::spawn(&McpServer {
        command: "flux-no-such-mcp-server-binary".into(),
        args: vec!["--nope".into()],
        env: Vec::new(),
    })
    .await
    .expect_err("a missing command must not succeed");

    let msg = err.to_string();
    assert!(
        msg.contains("flux-no-such-mcp-server-binary"),
        "error must name the command, got: {msg}"
    );
    assert!(msg.contains("PATH"), "error must hint at PATH, got: {msg}");
}

use flux_tool_client::{ToolCall, ToolResult};

async fn call(tool: &str, args: serde_json::Value) -> flux_core::error::Result<ToolResult> {
    let transport = McpTransport::spawn(&fixture()).await.expect("spawn");
    transport
        .call_tool(&ToolCall {
            name: tool.to_string(),
            args,
            context: None,
        })
        .await
}

#[tokio::test]
async fn calling_echo_upper_over_stdio_returns_the_fixed_shape() {
    let out = call("echo_upper", serde_json::json!({ "text": "hello mcp" }))
        .await
        .expect("call succeeds");
    assert_eq!(out.result["text"], "HELLO MCP");
    assert_eq!(out.result["content"][0]["text"], "HELLO MCP");
    assert!(out.result["structured_content"].is_null());
}

#[tokio::test]
async fn structured_output_reaches_the_structured_key() {
    let out = call("emit_structured", serde_json::json!({}))
        .await
        .expect("call succeeds");
    assert_eq!(out.result["structured_content"]["answer"], 42);
    assert_eq!(out.result["text"], "see structured");
}

#[tokio::test]
async fn a_tool_level_error_is_a_provider_error() {
    let err = call("always_fails", serde_json::json!({}))
        .await
        .expect_err("always_fails must fail");
    match err {
        flux_core::error::FluxError::Provider(m) => {
            assert!(m.contains("always_fails"), "names the tool: {m}");
            assert!(m.contains("boom"), "carries the server's message: {m}");
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn context_is_dropped_rather_than_failing_the_call() {
    let transport = McpTransport::spawn(&fixture()).await.expect("spawn");
    let out = transport
        .call_tool(&ToolCall {
            name: "echo_upper".into(),
            args: serde_json::json!({ "text": "ok" }),
            context: Some(serde_json::json!({ "tenant": "acme" })),
        })
        .await
        .expect("context must not fail the call");
    assert_eq!(out.result["text"], "OK");
}

/// `ps -p` is portable enough across macOS and Linux for a test, and avoids a
/// `libc` dev-dependency just to call `kill(pid, 0)`.
fn pid_is_alive(pid: u32) -> bool {
    std::process::Command::new("ps")
        .arg("-p")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn dropping_the_transport_kills_the_child() {
    let transport = McpTransport::spawn(&fixture()).await.expect("spawn");
    let pid = transport.child_pid().expect("child pid");
    assert!(pid_is_alive(pid), "child should be running before drop");

    drop(transport);

    // rmcp's cleanup kills the child from a `tokio::spawn`ed task, so the kill
    // is asynchronous and best-effort — poll rather than asserting immediately.
    let mut alive = true;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        if !pid_is_alive(pid) {
            alive = false;
            break;
        }
    }
    assert!(
        !alive,
        "child process {pid} outlived its transport (orphan leak)"
    );
}

/// What the fixture child sees for `var`, or `<unset>`.
async fn child_env(server: &McpServer, var: &str) -> String {
    let transport = McpTransport::spawn(server).await.expect("spawn");
    let out = transport
        .call_tool(&ToolCall {
            name: "read_env".into(),
            args: serde_json::json!({ "name": var }),
            context: None,
        })
        .await
        .expect("read_env succeeds");
    out.result["text"].as_str().expect("text").to_string()
}

// Cargo sets CARGO_MANIFEST_DIR for every test process, so it stands in for a
// secret Flux holds (an LLM key, a database URL) without mutating this
// process's environment under parallel tests.
const PARENT_VAR: &str = "CARGO_MANIFEST_DIR";

#[tokio::test]
async fn an_unlisted_variable_does_not_reach_the_server() {
    assert!(std::env::var(PARENT_VAR).is_ok(), "cargo sets {PARENT_VAR}");
    assert_eq!(child_env(&fixture(), PARENT_VAR).await, "<unset>");
}

#[tokio::test]
async fn a_listed_variable_reaches_the_server() {
    let server = McpServer {
        env: vec![PARENT_VAR.to_string()],
        ..fixture()
    };
    let expected = std::env::var(PARENT_VAR).expect("cargo sets it");
    assert_eq!(child_env(&server, PARENT_VAR).await, expected);
}

#[tokio::test]
async fn path_reaches_the_server_without_being_listed() {
    let expected = std::env::var("PATH").expect("PATH is set");
    assert_eq!(child_env(&fixture(), "PATH").await, expected);
}
