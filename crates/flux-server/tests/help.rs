//! `--help` must work from any directory, before config is loaded.

use std::process::Command;

#[test]
fn help_prints_usage_without_a_config_dir() {
    let out = Command::new(env!("CARGO_BIN_EXE_flux-server"))
        .arg("--help")
        .current_dir(std::env::temp_dir())
        .env_remove("FLUX_CONFIG_DIR")
        .output()
        .expect("run flux-server");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("usage: flux-server"), "stdout: {stdout}");
}
