# Working notes for Claude Code

This repository is the Capacity Planning Forecaster, package `headroom`: hierarchical
probabilistic demand forecasting with adaptive conformal intervals, MinT reconciliation,
named neural forecasters against statistical baselines, and a newsvendor staffing layer.
Build started 2026-09-12. The plan is in [PLAN.md](PLAN.md).

## Read first

- **[docs/state.md](docs/state.md): where the build is, what has been measured, what
  decision is open, and what to do next. Start here.**
- [README.md](README.md): what this is and the current result tables.
- [PLAN.md](PLAN.md): the design. Do not deviate from it silently; if something in it turns
  out wrong, change the plan in the same commit as the code and say why in the commit
  message.
- [docs/methods.md](docs/methods.md): the assumptions and every measured number with
  its provenance, especially conformal prediction under dependence.
- [docs/data.md](docs/data.md): the data, and what each cleaning decision cost.

## Engineering standard

- Python 3.13. Typed throughout; `mypy --strict` and `ruff` clean in CI.
- Tests that fail meaningfully: summing matrix and coherence, CRPS closed-form check,
  conformal tracking on a synthetic shift, block bootstrap ordering, newsvendor quantile,
  dashboard JSON schema.
- `pyproject.toml` with pinned major versions and a comment saying why for each pin.
- Docs ship in the same commit as the change.
- Never commit raw data, or anything from `.env`.

## Rules specific to this repository

- **Baselines first.** No neural result is reported without seasonal naive and the
  statistical models beside it, as skill.
- **Probabilistic scores are the result.** CRPS, pinball, coverage and width. MAE is a
  footnote at most.
- **State the conformal assumption** wherever coverage is shown.
- **Reconciled forecasts must be coherent** at every origin; the test enforces it.
- **The neural verdict is written whichever way it falls.** `docs/neural-verdict.md` says
  where the neural models did not earn their complexity.
- **Decision inputs are explicit and replaceable.** Demand-per-staff ratio and costs are a
  table, never buried constants.
- **Public data from other jurisdictions only.** No employer or provincial series.
- **Every reported number carries a confidence interval.**
- **Plain punctuation** in everything written here: no em-dashes or other typographic
  dashes, straight quotes only.

## What goes in the README

The README opens with the one-liner, the results tables and the honest limitation, before
any installation instructions. The report command fills the tables; do not hand-edit them.
Record one approach that was tried and rejected, with the evidence, once the work has
produced it.
