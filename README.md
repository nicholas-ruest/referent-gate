# referent-gate

`referent-gate` is a small, offline Rust CLI that stops an agent from silently
guessing which account, file, host, ticket, or other entity a consequential
action refers to.

It combines declared candidate priors with fresh evidence likelihoods, counts
only one observation from each independence group, applies impact-specific
probability and margin thresholds, and recommends the affordable probe with
the greatest expected information gain per unit cost. It never performs the
proposed action.

## Install

Requires stable Rust.

```console
cargo install --git https://github.com/nicholas-ruest/referent-gate
```

Or build from a checkout:

```console
cargo build --release
```

## Use

Assess a clear reference:

```console
cargo run -- \
  --scenario examples/clear.json \
  --policy examples/policy.toml
```

Output:

```text
ALLOW action=refund-ticket-4821 impact=medium
selected=account-a top=0.9989 margin=0.9979 independent_groups=2
posterior account-a=0.998927
posterior account-b=0.001073
```

An ambiguous high-impact reference returns exit code `2` and identifies the
best next question:

```console
cargo run -- \
  --scenario examples/ambiguous.json \
  --policy examples/policy.toml
```

```text
CLARIFY action=refund-ticket-4822 impact=high
...
next_probe=ask-invoice-suffix information_gain_bits=... score_per_cost=...
```

Use `--format json` for automation and `--verbose` for input/policy diagnostics.
Exit codes are `0` for allow, `2` for clarify/block, and `1` for invalid input
or configuration.

## Input contract

The scenario JSON declares:

- at least two candidates whose priors sum to `1.0`;
- timestamped observations with `P(observation | candidate)` likelihoods;
- an `independence_group` for each observation;
- optional probes with costs and exhaustive outcome distributions.

The policy TOML sets evidence freshness, the minimum number of independent
groups, the maximum probe cost, and low/medium/high posterior thresholds. See
[`examples/`](examples/) for complete inputs.

Fresh observations in the same independence group are deliberately collapsed
to the newest one. This prevents duplicated or correlated evidence from
artificially multiplying confidence.

## Verification

```console
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Limits

- Priors and likelihoods are supplied by the caller; the tool does not claim
  they are calibrated, independent, complete, or truthful.
- Independence groups are a declared contract, not statistically inferred.
- The Bayesian model is finite and closed-world. Unknown candidates must be
  represented explicitly if they matter.
- Information gain ranks declared probes; it does not ask questions, retrieve
  evidence, authorize an action, or replace IAM, policy, or human approval.
- A passing result is an ambiguity check for one frozen scenario, not proof
  that the action is correct or safe.

## License

MIT
