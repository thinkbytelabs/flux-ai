//! Regression test for the one gap the MCP transport work left open: nothing
//! proved that an **MCP-only** configuration — no gRPC endpoint at all — reaches
//! `connect_all` and yields a working transport. That was verified once by hand
//! against a running `flux-server`, which needs Redis and a live Postgres; this
//! pins it in CI with neither, and no network.
//!
//! It walks the same chain a binary walks: YAML on disk -> `Settings` ->
//! `has_backends()` -> the `McpServerCfg` -> `McpServer` mapping -> `connect_all`
//! -> a live MCP child process -> a real tool call.

use flux_tool_client::{ToolCall, ToolTransport};
use std::path::PathBuf;

/// Mirrors the private `mcp_servers()` helper that each of the three binaries
/// carries. The duplication is deliberate upstream — `flux-settings` must not
/// depend on `flux-tool-client`, so the two spec types stay separate — which is
/// exactly why the mapping deserves pinning somewhere: two fields, carried
/// across unchanged.
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

/// A config directory that deletes itself. `Settings::load_from_dir` reads
/// `base.yaml` plus a `<env>.yaml` overlay, so both are written.
struct TempConfigDir {
    path: PathBuf,
}

impl TempConfigDir {
    fn new(label: &str, tools_block: &str) -> TempConfigDir {
        // Unique per process and per test so parallel tests cannot collide.
        let path =
            std::env::temp_dir().join(format!("flux-mcp-cfg-{}-{label}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp config dir");
        // Every `Settings` field is `#[serde(default)]`, so only the block under
        // test needs writing.
        std::fs::write(path.join("base.yaml"), tools_block).expect("write base.yaml");
        std::fs::write(
            path.join("dev.yaml"),
            "telemetry:\n  log_format: \"pretty\"\n",
        )
        .expect("write dev.yaml");
        TempConfigDir { path }
    }

    fn load(&self) -> flux_settings::Settings {
        flux_settings::Settings::load_from_dir(&self.path, "dev").expect("load settings")
    }
}

impl Drop for TempConfigDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn fixture_command() -> String {
    env!("CARGO_BIN_EXE_mcp_fixture").to_string()
}

#[tokio::test]
async fn mcp_only_config_reaches_connect_all_and_serves_tools() {
    let cfg = TempConfigDir::new(
        "mcp-only",
        &format!("tools:\n  mcp:\n    - command: \"{}\"\n", fixture_command()),
    );
    let settings = cfg.load();

    // The state a binary gates on: no gRPC endpoint, but tools are configured.
    // Before `has_backends()` existed, this combination read as "no tool
    // support" and the MCP config was silently ignored.
    assert!(
        settings.tools.resolved_endpoints().is_empty(),
        "this test is only meaningful with zero gRPC endpoints"
    );
    assert!(
        settings.tools.has_backends(),
        "an MCP-only config must count as having tool backends"
    );
    assert_eq!(settings.tools.mcp.len(), 1);

    // The chain each binary runs: map the settings type to the transport type,
    // then hand both lists to connect_all with the gRPC side empty.
    let servers = mcp_servers(&settings);
    let transport = flux_tool_client::connect_all(&[], &servers)
        .await
        .expect("MCP-only config must build a transport");

    let manifest = transport.list_tools().await.expect("list tools");
    let mut names: Vec<&str> = manifest.tools.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["always_fails", "echo_upper", "emit_structured"]);

    // And it actually routes a call — a manifest alone would not prove the
    // transport is usable.
    let out = transport
        .call_tool(&ToolCall {
            name: "echo_upper".into(),
            args: serde_json::json!({ "text": "from config" }),
            context: None,
        })
        .await
        .expect("call routed through the config-derived transport");
    assert_eq!(out.result["text"], "FROM CONFIG");
}

#[tokio::test]
async fn config_with_neither_endpoint_nor_mcp_reports_no_backends() {
    // The other half of the gate: a binary must still preflight-fail here
    // rather than build an empty transport and die later on a tool call.
    let cfg = TempConfigDir::new("no-backends", "tools: {}\n");
    let settings = cfg.load();
    assert!(settings.tools.resolved_endpoints().is_empty());
    assert!(settings.tools.mcp.is_empty());
    assert!(!settings.tools.has_backends());
}

#[tokio::test]
async fn mcp_server_mapping_preserves_command_and_args_in_order() {
    // Pins the two-field mapping the three binaries duplicate. Argument order
    // matters: an MCP server launched with its args transposed is a different
    // program, and the failure would surface as a confusing spawn error.
    let cfg = TempConfigDir::new(
        "mapping",
        "tools:\n  mcp:\n    - command: npx\n      args: [\"-y\", \"@scope/server\", \"--flag\"]\n    - command: uvx\n",
    );
    let settings = cfg.load();
    let servers = mcp_servers(&settings);

    assert_eq!(servers.len(), 2);
    assert_eq!(servers[0].command, "npx");
    assert_eq!(servers[0].args, vec!["-y", "@scope/server", "--flag"]);
    assert_eq!(servers[1].command, "uvx");
    assert!(
        servers[1].args.is_empty(),
        "a server with no args maps to an empty vec, not a placeholder"
    );
}

#[tokio::test]
async fn two_mcp_servers_exporting_the_same_tool_is_a_config_error() {
    // Same fixture twice: the duplicate-name check must fire from a
    // config-derived server list, not just from hand-built specs.
    let cfg = TempConfigDir::new(
        "dup",
        &format!(
            "tools:\n  mcp:\n    - command: \"{c}\"\n    - command: \"{c}\"\n",
            c = fixture_command()
        ),
    );
    let settings = cfg.load();
    assert!(settings.tools.has_backends());

    let servers = mcp_servers(&settings);
    assert_eq!(servers.len(), 2);

    // `expect_err` is unavailable here: it needs the Ok type to be `Debug`, and
    // `AnyTransport` deliberately has none (`GrpcTransport` has none either).
    match flux_tool_client::connect_all(&[], &servers).await {
        Ok(_) => panic!("two servers exporting echo_upper must not both register"),
        Err(flux_core::error::FluxError::Config(m)) => {
            assert!(m.contains("echo_upper"), "names the clashing tool: {m}");
        }
        Err(other) => panic!("expected Config error, got {other:?}"),
    }
}
