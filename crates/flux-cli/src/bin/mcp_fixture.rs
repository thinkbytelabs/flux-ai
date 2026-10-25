//! A minimal MCP server over stdio, used as a hermetic test fixture for Flux's
//! MCP tool transport. NOT an example of good MCP server design and NOT a
//! supported binary — it exists so `mcp_transport.rs` and
//! `mcp_tool_stage_e2e.rs` can drive a real child process with no network and
//! no node/python toolchain.
//!
//! Protocol: newline-delimited JSON-RPC 2.0. One JSON object per line on stdin,
//! one response line per request that carries an `id`. Nothing else may ever be
//! written to stdout — logs go to stderr.

use serde_json::{json, Value};
use std::io::{BufRead, Write};

fn tool_specs() -> Value {
    json!([
        {
            "name": "echo_upper",
            "description": "Uppercase the given text",
            "inputSchema": {
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }
        },
        {
            "name": "emit_structured",
            "description": "Return structured content alongside text",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "always_fails",
            "description": "Always reports a tool-level error",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "read_env",
            "description": "Report one environment variable of this process",
            "inputSchema": {
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }
        }
    ])
}

fn call_tool(params: &Value) -> Value {
    let name = params["name"].as_str().unwrap_or_default();
    let args = &params["arguments"];
    match name {
        "echo_upper" => {
            let text = args["text"].as_str().unwrap_or_default();
            json!({
                "content": [{ "type": "text", "text": text.to_uppercase() }],
                "isError": false
            })
        }
        "emit_structured" => json!({
            "content": [{ "type": "text", "text": "see structured" }],
            "structuredContent": { "answer": 42 },
            "isError": false
        }),
        "always_fails" => json!({
            "content": [{ "type": "text", "text": "boom" }],
            "isError": true
        }),
        // Lets tests see which variables Flux passed to the child.
        "read_env" => {
            let var = args["name"].as_str().unwrap_or_default();
            let value = std::env::var(var).unwrap_or_else(|_| "<unset>".to_string());
            json!({
                "content": [{ "type": "text", "text": value }],
                "isError": false
            })
        }
        other => json!({
            "content": [{ "type": "text", "text": format!("unknown tool: {other}") }],
            "isError": true
        }),
    }
}

/// `Ok(result)` becomes a JSON-RPC result; `Err(())` becomes method-not-found.
fn dispatch(method: &str, params: &Value) -> std::result::Result<Value, ()> {
    match method {
        // Echo the client's protocol version back so any rmcp release accepts
        // this fixture without the version being pinned here.
        "initialize" => Ok(json!({
            "protocolVersion": params["protocolVersion"],
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "flux-mcp-fixture", "version": "0.1.0" }
        })),
        "tools/list" => Ok(json!({ "tools": tool_specs() })),
        "tools/call" => Ok(call_tool(params)),
        _ => Err(()),
    }
}

fn main() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    // Completes the `initialize` handshake, then answers nothing ever again
    // while staying alive and readable. This is the shape of hang that a
    // handshake-only timeout cannot catch: the child is healthy, the pipe is
    // open, and the next request simply never gets a reply.
    let silent_after_initialize = std::env::args().any(|a| a == "--silent-after-initialize");

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            eprintln!("mcp_fixture: ignoring unparseable line");
            continue;
        };
        // No id means a notification: process nothing, answer nothing.
        let Some(id) = req.get("id").cloned() else {
            continue;
        };
        let method = req["method"].as_str().unwrap_or_default();
        if silent_after_initialize && method != "initialize" {
            continue;
        }
        let response = match dispatch(method, &req["params"]) {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(()) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("method not found: {method}") }
            }),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
