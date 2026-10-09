#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::Command;

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

fn command(scenario: &str, format: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_referent-gate"))
        .args([
            "--scenario",
            example(scenario).to_str().unwrap(),
            "--policy",
            example("policy.toml").to_str().unwrap(),
            "--format",
            format,
        ])
        .output()
        .unwrap()
}

#[test]
fn clear_scenario_exits_zero() {
    let output = command("clear.json", "text");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("ALLOW action=refund-ticket-4821"));
    assert!(stdout.contains("selected=account-a"));
}

#[test]
fn ambiguous_scenario_exits_two_with_probe() {
    let output = command("ambiguous.json", "text");
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("CLARIFY action=refund-ticket-4822"));
    assert!(stdout.contains("next_probe=ask-invoice-suffix"));
}

#[test]
fn json_report_is_machine_readable() {
    let output = command("ambiguous.json", "json");
    assert_eq!(output.status.code(), Some(2));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["verdict"], "clarify");
    assert_eq!(report["recommendation"]["probe"], "ask-invoice-suffix");
}
