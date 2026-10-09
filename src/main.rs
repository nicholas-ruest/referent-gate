#![forbid(unsafe_code)]

use std::{fs, path::PathBuf, process::ExitCode};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use referent_gate::{Policy, Scenario, Verdict, evaluate};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Scenario JSON containing candidates, observations, and optional probes.
    #[arg(long)]
    scenario: PathBuf,

    /// Gate policy in TOML format.
    #[arg(long)]
    policy: PathBuf,

    /// Report format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Emit input and policy diagnostics to stderr.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Format {
    Text,
    Json,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    let scenario_text = fs::read_to_string(&cli.scenario)
        .with_context(|| format!("could not read scenario {}", cli.scenario.display()))?;
    let policy_text = fs::read_to_string(&cli.policy)
        .with_context(|| format!("could not read policy {}", cli.policy.display()))?;
    let scenario: Scenario = serde_json::from_str(&scenario_text)
        .with_context(|| format!("invalid scenario JSON in {}", cli.scenario.display()))?;
    let policy: Policy = toml::from_str(&policy_text)
        .with_context(|| format!("invalid policy TOML in {}", cli.policy.display()))?;

    if cli.verbose {
        eprintln!(
            "loaded action={} impact={} candidates={} observations={} probes={}",
            scenario.action_id,
            scenario.impact,
            scenario.candidates.len(),
            scenario.observations.len(),
            scenario.probes.len()
        );
        eprintln!(
            "policy max_age={}s min_groups={} max_probe_cost={}",
            policy.max_evidence_age_seconds, policy.min_independent_groups, policy.max_probe_cost
        );
    }

    let report = evaluate(&scenario, &policy).context("scenario evaluation failed")?;
    match cli.format {
        Format::Text => print!("{}", report.to_text()),
        Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }

    Ok(if report.verdict == Verdict::Allow {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}
