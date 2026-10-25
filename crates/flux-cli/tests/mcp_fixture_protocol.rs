//! Proves the MCP fixture server speaks the protocol, using raw JSON-RPC over
//! its stdin/stdout. Deliberately does not use rmcp: this is the independent
//! implementation the rmcp client is tested against.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Spawn the fixture, send each request line, and collect one response line per
/// request that carries an `id`. Notifications (no `id`) get no response.
fn exchange(requests: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mcp_fixture"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn mcp fixture");

    let expected = requests.iter().filter(|r| r.get("id").is_some()).count();

    {
        let stdin = child.stdin.as_mut().expect("fixture stdin");
        for req in requests {
            writeln!(stdin, "{req}").expect("write request");
        }
        stdin.flush().expect("flush");
    }
    // Drop stdin so the fixture's read loop sees EOF and exits.
    drop(child.stdin.take());

    let stdout = child.stdout.take().expect("fixture stdout");
    let mut responses = Vec::new();
    for line in BufReader::new(stdout).lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        responses.push(serde_json::from_str(&line).expect("response is JSON"));
    }
    let _ = child.wait();
    assert_eq!(
        responses.len(),
        expected,
        "one response per request with an id"
    );
    responses
}

fn initialize() -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.0.0" }
        }
    })
}

#[test]
fn initialize_echoes_the_clients_protocol_version() {
    let out = exchange(&[initialize()]);
    assert_eq!(out[0]["jsonrpc"], "2.0");
    assert_eq!(out[0]["id"], 1);
    assert_eq!(out[0]["result"]["protocolVersion"], "2025-06-18");
    assert!(out[0]["result"]["capabilities"]["tools"].is_object());
    assert_eq!(out[0]["result"]["serverInfo"]["name"], "flux-mcp-fixture");
}

#[test]
fn notifications_get_no_response() {
    let out = exchange(&[
        initialize(),
        serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    ]);
    assert_eq!(out.len(), 1, "the notification must not be answered");
}

#[test]
fn tools_list_advertises_four_tools_with_schemas() {
    let out = exchange(&[
        initialize(),
        serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
    ]);
    let tools = out[1]["result"]["tools"].as_array().expect("tools array");
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(
        names,
        vec!["echo_upper", "emit_structured", "always_fails", "read_env"]
    );
    assert_eq!(tools[0]["inputSchema"]["type"], "object");
    assert_eq!(tools[0]["inputSchema"]["required"][0], "text");
}

#[test]
fn echo_upper_uppercases_its_argument() {
    let out = exchange(&[
        initialize(),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "echo_upper", "arguments": { "text": "hello mcp" } }
        }),
    ]);
    assert_eq!(out[1]["result"]["content"][0]["type"], "text");
    assert_eq!(out[1]["result"]["content"][0]["text"], "HELLO MCP");
    assert_eq!(out[1]["result"]["isError"], false);
}

#[test]
fn emit_structured_sets_structured_content() {
    let out = exchange(&[
        initialize(),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "emit_structured", "arguments": {} }
        }),
    ]);
    assert_eq!(out[1]["result"]["structuredContent"]["answer"], 42);
    assert_eq!(out[1]["result"]["content"][0]["text"], "see structured");
}

#[test]
fn always_fails_sets_is_error() {
    let out = exchange(&[
        initialize(),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "always_fails", "arguments": {} }
        }),
    ]);
    assert_eq!(out[1]["result"]["isError"], true);
    assert_eq!(out[1]["result"]["content"][0]["text"], "boom");
}

#[test]
fn unknown_method_returns_a_jsonrpc_error() {
    let out = exchange(&[
        initialize(),
        serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "resources/list" }),
    ]);
    assert_eq!(out[1]["error"]["code"], -32601);
    assert!(out[1].get("result").is_none());
}
