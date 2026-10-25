//! `--help` must work from any directory, before config is loaded.

use std::process::Command;

fn help(bin: &str) -> String {
    let out = Command::new(bin)
        .arg("--help")
        .current_dir(std::env::temp_dir())
        .env_remove("FLUX_CONFIG_DIR")
        .output()
        .expect("run binary");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn run_agent_help_prints_usage_without_a_config_dir() {
    assert!(help(env!("CARGO_BIN_EXE_run_agent")).contains("usage: run_agent"));
}

#[test]
fn run_workflow_help_prints_usage_without_a_config_dir() {
    assert!(help(env!("CARGO_BIN_EXE_run_workflow")).contains("usage: run_workflow"));
}
