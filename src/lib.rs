#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write};

use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const PROBABILITY_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub action_id: String,
    pub as_of: DateTime<Utc>,
    pub impact: Impact,
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub observations: Vec<Observation>,
    #[serde(default)]
    pub probes: Vec<Probe>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: String,
    pub prior: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: String,
    pub observed_at: DateTime<Utc>,
    pub independence_group: String,
    pub likelihoods: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    pub id: String,
    pub cost: f64,
    pub outcomes: Vec<ProbeOutcome>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeOutcome {
    pub id: String,
    pub likelihoods: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub max_evidence_age_seconds: u64,
    pub min_independent_groups: usize,
    pub max_probe_cost: f64,
    pub thresholds: ThresholdPolicy,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdPolicy {
    pub low: Thresholds,
    pub medium: Thresholds,
    pub high: Thresholds,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    pub min_top_probability: f64,
    pub min_margin: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Impact {
    Low,
    Medium,
    High,
}

impl fmt::Display for Impact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        })
    }
}

impl Policy {
    fn thresholds_for(&self, impact: Impact) -> Thresholds {
        match impact {
            Impact::Low => self.thresholds.low,
            Impact::Medium => self.thresholds.medium,
            Impact::High => self.thresholds.high,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Allow,
    Clarify,
    Block,
}

impl fmt::Display for Verdict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Allow => "ALLOW",
            Self::Clarify => "CLARIFY",
            Self::Block => "BLOCK",
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub schema_version: u8,
    pub action_id: String,
    pub as_of: DateTime<Utc>,
    pub impact: Impact,
    pub verdict: Verdict,
    pub selected_candidate: Option<String>,
    pub top_probability: f64,
    pub margin: f64,
    pub independent_groups: usize,
    pub used_observations: Vec<String>,
    pub ignored_observations: Vec<String>,
    pub posterior: Vec<PosteriorEntry>,
    pub findings: Vec<Finding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<Recommendation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PosteriorEntry {
    pub candidate: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Recommendation {
    pub probe: String,
    pub expected_information_gain_bits: f64,
    pub score_per_cost: f64,
}

impl Report {
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut output = String::new();
        let selected = self.selected_candidate.as_deref().unwrap_or("none");
        writeln!(
            output,
            "{} action={} impact={}",
            self.verdict, self.action_id, self.impact
        )
        .expect("writing to String cannot fail");
        writeln!(
            output,
            "selected={} top={:.4} margin={:.4} independent_groups={}",
            selected, self.top_probability, self.margin, self.independent_groups
        )
        .expect("writing to String cannot fail");
        for entry in &self.posterior {
            writeln!(
                output,
                "posterior {}={:.6}",
                entry.candidate, entry.probability
            )
            .expect("writing to String cannot fail");
        }
        for finding in &self.findings {
            writeln!(output, "finding {}: {}", finding.code, finding.message)
                .expect("writing to String cannot fail");
        }
        if let Some(recommendation) = &self.recommendation {
            writeln!(
                output,
                "next_probe={} information_gain_bits={:.6} score_per_cost={:.6}",
                recommendation.probe,
                recommendation.expected_information_gain_bits,
                recommendation.score_per_cost
            )
            .expect("writing to String cannot fail");
        }
        output
    }
}

/// Evaluate one proposed action without performing it.
///
/// Fresh observations are collapsed to one per declared independence group so
/// callers cannot inflate confidence by submitting correlated evidence twice.
pub fn evaluate(scenario: &Scenario, policy: &Policy) -> Result<Report> {
    validate(scenario, policy)?;
    let candidate_ids: Vec<&str> = scenario
        .candidates
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();

    let mut findings = Vec::new();
    let mut ignored = Vec::new();
    let mut freshest_by_group: BTreeMap<&str, &Observation> = BTreeMap::new();

    for observation in &scenario.observations {
        let age = scenario
            .as_of
            .signed_duration_since(observation.observed_at)
            .num_seconds();
        if age > policy.max_evidence_age_seconds as i64 {
            ignored.push(observation.id.clone());
            findings.push(Finding {
                code: "stale_evidence_ignored",
                message: format!(
                    "observation {} is {}s old; maximum is {}s",
                    observation.id, age, policy.max_evidence_age_seconds
                ),
            });
            continue;
        }

        match freshest_by_group.get(observation.independence_group.as_str()) {
            Some(current) if current.observed_at >= observation.observed_at => {
                ignored.push(observation.id.clone());
                findings.push(Finding {
                    code: "correlated_evidence_collapsed",
                    message: format!(
                        "observation {} was not counted because group {} already uses newer observation {}",
                        observation.id, observation.independence_group, current.id
                    ),
                });
            }
            Some(current) => {
                ignored.push(current.id.clone());
                findings.push(Finding {
                    code: "correlated_evidence_collapsed",
                    message: format!(
                        "observation {} replaced older observation {} in group {}",
                        observation.id, current.id, observation.independence_group
                    ),
                });
                freshest_by_group.insert(&observation.independence_group, observation);
            }
            None => {
                freshest_by_group.insert(&observation.independence_group, observation);
            }
        }
    }

    let mut log_weights: Vec<f64> = scenario
        .candidates
        .iter()
        .map(|candidate| candidate.prior.ln())
        .collect();
    for observation in freshest_by_group.values() {
        for (index, id) in candidate_ids.iter().enumerate() {
            log_weights[index] += observation.likelihoods[*id].ln();
        }
    }
    let probabilities = normalize_log_weights(&log_weights)?;
    let mut posterior: Vec<PosteriorEntry> = scenario
        .candidates
        .iter()
        .zip(probabilities.iter())
        .map(|(candidate, probability)| PosteriorEntry {
            candidate: candidate.id.clone(),
            probability: *probability,
        })
        .collect();
    posterior.sort_by(|left, right| {
        right
            .probability
            .partial_cmp(&left.probability)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.candidate.cmp(&right.candidate))
    });

    let top_probability = posterior[0].probability;
    let runner_up = posterior.get(1).map_or(0.0, |entry| entry.probability);
    let margin = top_probability - runner_up;
    let thresholds = policy.thresholds_for(scenario.impact);
    let groups_ok = freshest_by_group.len() >= policy.min_independent_groups;
    let top_ok = top_probability >= thresholds.min_top_probability;
    let margin_ok = margin >= thresholds.min_margin;

    if !groups_ok {
        findings.push(Finding {
            code: "insufficient_independent_evidence",
            message: format!(
                "{} independent groups available; policy requires {}",
                freshest_by_group.len(),
                policy.min_independent_groups
            ),
        });
    }
    if !top_ok {
        findings.push(Finding {
            code: "top_probability_below_threshold",
            message: format!(
                "top probability {top_probability:.4} is below {:.4}",
                thresholds.min_top_probability
            ),
        });
    }
    if !margin_ok {
        findings.push(Finding {
            code: "posterior_margin_below_threshold",
            message: format!(
                "posterior margin {margin:.4} is below {:.4}",
                thresholds.min_margin
            ),
        });
    }

    let recommendation = if groups_ok && top_ok && margin_ok {
        None
    } else {
        best_probe(scenario, policy, &probabilities)?
    };
    let verdict = if groups_ok && top_ok && margin_ok {
        Verdict::Allow
    } else if recommendation.is_some() {
        Verdict::Clarify
    } else {
        Verdict::Block
    };

    let mut used_observations: Vec<String> = freshest_by_group
        .values()
        .map(|observation| observation.id.clone())
        .collect();
    used_observations.sort();
    ignored.sort();

    Ok(Report {
        schema_version: 1,
        action_id: scenario.action_id.clone(),
        as_of: scenario.as_of,
        impact: scenario.impact,
        verdict,
        selected_candidate: (verdict == Verdict::Allow).then(|| posterior[0].candidate.clone()),
        top_probability,
        margin,
        independent_groups: freshest_by_group.len(),
        used_observations,
        ignored_observations: ignored,
        posterior,
        findings,
        recommendation,
    })
}

fn validate(scenario: &Scenario, policy: &Policy) -> Result<()> {
    if scenario.action_id.trim().is_empty() {
        bail!("action_id must not be empty");
    }
    if scenario.candidates.len() < 2 {
        bail!("at least two candidates are required");
    }
    if policy.max_evidence_age_seconds == 0 {
        bail!("max_evidence_age_seconds must be greater than zero");
    }
    if policy.min_independent_groups == 0 {
        bail!("min_independent_groups must be greater than zero");
    }
    if !policy.max_probe_cost.is_finite() || policy.max_probe_cost <= 0.0 {
        bail!("max_probe_cost must be finite and greater than zero");
    }
    for thresholds in [
        policy.thresholds.low,
        policy.thresholds.medium,
        policy.thresholds.high,
    ] {
        check_unit_interval("min_top_probability", thresholds.min_top_probability, false)?;
        check_unit_interval("min_margin", thresholds.min_margin, true)?;
    }

    let mut ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for candidate in &scenario.candidates {
        if candidate.id.trim().is_empty() || !ids.insert(candidate.id.as_str()) {
            bail!("candidate IDs must be non-empty and unique");
        }
        check_unit_interval("candidate prior", candidate.prior, false)?;
        prior_sum += candidate.prior;
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        bail!("candidate priors must sum to 1.0; got {prior_sum}");
    }

    let mut observation_ids = BTreeSet::new();
    for observation in &scenario.observations {
        if observation.id.trim().is_empty() || !observation_ids.insert(observation.id.as_str()) {
            bail!("observation IDs must be non-empty and unique");
        }
        if observation.independence_group.trim().is_empty() {
            bail!(
                "observation {} has an empty independence_group",
                observation.id
            );
        }
        if observation.observed_at > scenario.as_of {
            bail!(
                "observation {} is later than scenario as_of",
                observation.id
            );
        }
        validate_likelihood_keys(&ids, &observation.likelihoods, &observation.id)?;
    }

    let mut probe_ids = BTreeSet::new();
    for probe in &scenario.probes {
        if probe.id.trim().is_empty() || !probe_ids.insert(probe.id.as_str()) {
            bail!("probe IDs must be non-empty and unique");
        }
        if !probe.cost.is_finite() || probe.cost <= 0.0 {
            bail!(
                "probe {} cost must be finite and greater than zero",
                probe.id
            );
        }
        if probe.outcomes.len() < 2 {
            bail!("probe {} requires at least two outcomes", probe.id);
        }
        let mut outcome_ids = BTreeSet::new();
        for outcome in &probe.outcomes {
            if outcome.id.trim().is_empty() || !outcome_ids.insert(outcome.id.as_str()) {
                bail!(
                    "outcome IDs for probe {} must be non-empty and unique",
                    probe.id
                );
            }
            validate_likelihood_keys(&ids, &outcome.likelihoods, &outcome.id)?;
        }
        for candidate_id in &ids {
            let sum: f64 = probe
                .outcomes
                .iter()
                .map(|outcome| outcome.likelihoods[*candidate_id])
                .sum();
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                bail!(
                    "probe {} outcome likelihoods for candidate {} must sum to 1.0; got {}",
                    probe.id,
                    candidate_id,
                    sum
                );
            }
        }
    }
    Ok(())
}

fn validate_likelihood_keys(
    candidate_ids: &BTreeSet<&str>,
    likelihoods: &BTreeMap<String, f64>,
    owner_id: &str,
) -> Result<()> {
    let keys: BTreeSet<&str> = likelihoods.keys().map(String::as_str).collect();
    if &keys != candidate_ids {
        bail!("likelihood keys for {owner_id} must exactly match candidate IDs");
    }
    for value in likelihoods.values() {
        check_unit_interval("likelihood", *value, false)?;
    }
    Ok(())
}

fn check_unit_interval(label: &str, value: f64, allow_zero: bool) -> Result<()> {
    let lower_ok = if allow_zero {
        value >= 0.0
    } else {
        value > 0.0
    };
    if !value.is_finite() || !lower_ok || value > 1.0 {
        bail!(
            "{label} must be finite and in {}",
            if allow_zero { "[0, 1]" } else { "(0, 1]" }
        );
    }
    Ok(())
}

fn normalize_log_weights(log_weights: &[f64]) -> Result<Vec<f64>> {
    let maximum = log_weights
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| anyhow!("cannot normalize an empty candidate set"))?;
    let weights: Vec<f64> = log_weights
        .iter()
        .map(|weight| (weight - maximum).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        bail!("posterior normalization failed");
    }
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

fn best_probe(
    scenario: &Scenario,
    policy: &Policy,
    current: &[f64],
) -> Result<Option<Recommendation>> {
    let current_entropy = entropy(current);
    let mut recommendations = Vec::new();
    for probe in &scenario.probes {
        if probe.cost > policy.max_probe_cost {
            continue;
        }
        let mut expected_entropy = 0.0;
        for outcome in &probe.outcomes {
            let joint: Vec<f64> = scenario
                .candidates
                .iter()
                .enumerate()
                .map(|(index, candidate)| current[index] * outcome.likelihoods[&candidate.id])
                .collect();
            let outcome_probability: f64 = joint.iter().sum();
            if outcome_probability > 0.0 {
                let conditional: Vec<f64> = joint
                    .iter()
                    .map(|value| value / outcome_probability)
                    .collect();
                expected_entropy += outcome_probability * entropy(&conditional);
            }
        }
        let information_gain = (current_entropy - expected_entropy).max(0.0);
        recommendations.push(Recommendation {
            probe: probe.id.clone(),
            expected_information_gain_bits: information_gain,
            score_per_cost: information_gain / probe.cost,
        });
    }
    recommendations.sort_by(|left, right| {
        right
            .score_per_cost
            .partial_cmp(&left.score_per_cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.probe.cmp(&right.probe))
    });
    Ok(recommendations.into_iter().next())
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn base_policy() -> Policy {
        Policy {
            max_evidence_age_seconds: 3_600,
            min_independent_groups: 2,
            max_probe_cost: 5.0,
            thresholds: ThresholdPolicy {
                low: Thresholds {
                    min_top_probability: 0.7,
                    min_margin: 0.3,
                },
                medium: Thresholds {
                    min_top_probability: 0.8,
                    min_margin: 0.5,
                },
                high: Thresholds {
                    min_top_probability: 0.95,
                    min_margin: 0.8,
                },
            },
        }
    }

    fn at(minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 8, 21, minute, 0)
            .single()
            .unwrap()
    }

    fn likelihoods(a: f64, b: f64) -> BTreeMap<String, f64> {
        BTreeMap::from([("account-a".into(), a), ("account-b".into(), b)])
    }

    fn base_scenario() -> Scenario {
        Scenario {
            action_id: "issue-refund".into(),
            as_of: at(30),
            impact: Impact::Medium,
            candidates: vec![
                Candidate {
                    id: "account-a".into(),
                    prior: 0.5,
                },
                Candidate {
                    id: "account-b".into(),
                    prior: 0.5,
                },
            ],
            observations: vec![
                Observation {
                    id: "crm-email".into(),
                    observed_at: at(25),
                    independence_group: "crm".into(),
                    likelihoods: likelihoods(0.95, 0.05),
                },
                Observation {
                    id: "invoice-code".into(),
                    observed_at: at(29),
                    independence_group: "billing".into(),
                    likelihoods: likelihoods(0.9, 0.1),
                },
            ],
            probes: Vec::new(),
        }
    }

    #[test]
    fn allows_clear_referent() {
        let report = evaluate(&base_scenario(), &base_policy()).unwrap();
        assert_eq!(report.verdict, Verdict::Allow);
        assert_eq!(report.selected_candidate.as_deref(), Some("account-a"));
        assert!(report.top_probability > 0.99);
        assert_eq!(report.independent_groups, 2);
    }

    #[test]
    fn impact_changes_threshold() {
        let mut scenario = base_scenario();
        scenario.observations[0].likelihoods = likelihoods(0.8, 0.2);
        scenario.observations[1].likelihoods = likelihoods(0.7, 0.3);
        scenario.impact = Impact::Low;
        assert_eq!(
            evaluate(&scenario, &base_policy()).unwrap().verdict,
            Verdict::Allow
        );
        scenario.impact = Impact::High;
        assert_eq!(
            evaluate(&scenario, &base_policy()).unwrap().verdict,
            Verdict::Block
        );
    }

    #[test]
    fn ignores_stale_evidence() {
        let mut scenario = base_scenario();
        scenario.observations[0].observed_at = at(0);
        let mut policy = base_policy();
        policy.max_evidence_age_seconds = 600;
        let report = evaluate(&scenario, &policy).unwrap();
        assert_eq!(report.verdict, Verdict::Block);
        assert_eq!(report.ignored_observations, vec!["crm-email"]);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.code == "stale_evidence_ignored")
        );
    }

    #[test]
    fn collapses_correlated_observations_to_newest() {
        let mut scenario = base_scenario();
        scenario.observations.push(Observation {
            id: "crm-email-new".into(),
            observed_at: at(28),
            independence_group: "crm".into(),
            likelihoods: likelihoods(0.1, 0.9),
        });
        let report = evaluate(&scenario, &base_policy()).unwrap();
        assert!(report.ignored_observations.contains(&"crm-email".into()));
        assert!(report.used_observations.contains(&"crm-email-new".into()));
        assert_eq!(report.independent_groups, 2);
    }

    #[test]
    fn recommends_probe_by_information_gain_per_cost() {
        let mut scenario = base_scenario();
        scenario.observations.clear();
        scenario.probes = vec![
            Probe {
                id: "weak-cheap".into(),
                cost: 1.0,
                outcomes: vec![
                    ProbeOutcome {
                        id: "yes".into(),
                        likelihoods: likelihoods(0.6, 0.4),
                    },
                    ProbeOutcome {
                        id: "no".into(),
                        likelihoods: likelihoods(0.4, 0.6),
                    },
                ],
            },
            Probe {
                id: "strong".into(),
                cost: 2.0,
                outcomes: vec![
                    ProbeOutcome {
                        id: "a".into(),
                        likelihoods: likelihoods(0.95, 0.05),
                    },
                    ProbeOutcome {
                        id: "b".into(),
                        likelihoods: likelihoods(0.05, 0.95),
                    },
                ],
            },
        ];
        let report = evaluate(&scenario, &base_policy()).unwrap();
        assert_eq!(report.verdict, Verdict::Clarify);
        assert_eq!(report.recommendation.unwrap().probe, "strong");
    }

    #[test]
    fn rejects_priors_that_do_not_sum_to_one() {
        let mut scenario = base_scenario();
        scenario.candidates[0].prior = 0.6;
        let error = evaluate(&scenario, &base_policy()).unwrap_err();
        assert!(error.to_string().contains("priors must sum"));
    }

    #[test]
    fn rejects_probe_outcomes_that_are_not_distributions() {
        let mut scenario = base_scenario();
        scenario.probes.push(Probe {
            id: "broken".into(),
            cost: 1.0,
            outcomes: vec![
                ProbeOutcome {
                    id: "yes".into(),
                    likelihoods: likelihoods(0.8, 0.8),
                },
                ProbeOutcome {
                    id: "no".into(),
                    likelihoods: likelihoods(0.8, 0.8),
                },
            ],
        });
        let error = evaluate(&scenario, &base_policy()).unwrap_err();
        assert!(error.to_string().contains("must sum to 1.0"));
    }
}
