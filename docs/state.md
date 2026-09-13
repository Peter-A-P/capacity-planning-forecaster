# State of the build

**Last updated: 2026-09-13.** Read this first if you are picking the project up. It says
what exists, what has been measured, what decision is open, and what to do next.

`PLAN.md` is the design and takes precedence. `docs/methods.md` has every measured number
with its provenance. `docs/data.md` describes the data and what each cleaning decision
cost. This file is the working state, and it goes stale; the other three do not.

---

## Where the build is

Started 2026-09-12, moved forward from the Jul 2027 slot (`PLAN.md` header, and the plan
repository at rev. 5). Two-week build; this is day 2.

**142 tests, `ruff` and `mypy --strict` clean.** Run `uv run pytest -q`; add `--run-slow`
for the tests that fit a real model, `--run-network` for the ones that fetch.

| File | Tests | Covers |
|---|---:|---|
| `test_backtest.py` | 38 | Origins and look-ahead, seasonal naive, the runner, checkpointing |
| `test_score.py` | 34 | CRPS against the closed form, pinball, coverage, width, skill, block bootstrap |
| `test_conformal.py` | 20 | The feedback rule, the conformal quantile, split, adaptive, aggregated |
| `test_data.py` | 20 | The loader, the borough judgement call, checks, the calendar |
| `test_stats_models.py` | 16 | The StatsForecast wrapper and the batched path |
| `test_hierarchy.py` | 14 | The summing matrix and coherence |

### Built and measured

| Area | Module | State |
|---|---|---|
| Data | `headroom.data.nyc_ems`, `checks`, `calendar` | Done. 29,977,935 incidents loaded, checked, documented |
| Hierarchy | `headroom.hierarchy` | Done. 37 nodes, coherence error 0.0 |
| Scores | `headroom.score` | Done. CRPS, pinball, coverage, width, skill, block bootstrap |
| Backtest | `headroom.backtest` | Done. 964 weekly origins, resumable checkpointing |
| Baseline | `headroom.models.baselines` | Done. Seasonal naive with per-horizon empirical residual quantiles |
| Conformal | `headroom.conformal` | Done. Split, adaptive (ACI), aggregated (AgACI) |
| Statistical models | `headroom.models.stats` | **Wrapper built and tested; the backtest has not been run.** See the open decision |
| CLI | `headroom.cli` | Done for what exists |

### Not built yet

- **Neural models** (N-HiTS, PatchTST). `PLAN.md` section 2.7. Week 2. The refit
  schedule is set from a measured single-fit time on an idle machine; weekly refits are
  unlikely to be affordable on CPU.
- **TimesFM, zero-shot** (added to the plan 2026-09-13). `PLAN.md` section 2.7a. Week 2.
  First check it installs under Python 3.13. Its pretraining postdates most origins, so
  it is scored separately on a clean window after its pretraining ends; read 2.7a before
  reporting any TimesFM number. Version pinned to 2.5 and the corpus researched
  (`docs/methods.md`, TimesFM section): clean window from 2023-12-01, cross-check from
  2025-09-15, intervals from conformal around its median.
- **Reconciliation** (MinT, probabilistic, coherence verified at every origin).
  `PLAN.md` section 2.5. Week 2. The summing matrix it needs is already in
  `headroom.hierarchy.spec` and tested.
- **Decision layer** (newsvendor, staffing, realised cost against an oracle).
  `PLAN.md` section 2.6. Week 2.
- **Charts and the report command.** The README results tables are still empty and must
  be filled by the report command, never by hand (`CLAUDE.md`).
- **Dashboard.** Deliberately held until project 01 builds the static decision-app pattern
  in its week 7 (Oct 19 to 25 2026), so 08 reuses it rather than inventing it. Peter's
  call, recorded in `PLAN.md`.
- **NHS England dataset.** Optional and first to drop (`PLAN.md` section 5).

---

## Decided: weekly origins for the statistical models

**Decided, and running.** ETS, Theta and MSTL at weekly origins (964) started
2026-09-13 13:20, logging to `backtest/out/stats-weekly.log`. Early origins ran at about
21 seconds each for all three together, projecting about 5.5 hours, well under the
estimates below. Record the measured total in `docs/methods.md` when it finishes, and
do not time anything else while it runs. The estimates and reasoning below are kept as
the record of how the decision was made.

The statistical models have to be back-tested over the record, and that is hours of CPU.
The measured cost, on an **idle** machine, 37 series, 1,095-day window, 12 cores:

| Model | Seconds per origin |
|---|---:|
| ETS | 33 |
| Theta | 34 |
| MSTL | 42 |
| AutoARIMA | 166 |
| All four in one call | 190 |

Projected totals for ETS + Theta + MSTL:

| Schedule | Origins | Estimated |
|---|---:|---:|
| Weekly (matches the baseline and conformal runs) | 964 | ~20 h |
| Fortnightly | 482 | ~15-19 h |
| **Monthly** | 241 | **~7.5-9.5 h** |

AutoARIMA roughly doubles whichever of those is chosen.

**The recommendation on the table was monthly, run overnight on an idle machine, with the
loss of shift resolution reported honestly.** Monthly origins put only 1 to 2 origins
inside the six-week March 2020 surge, against about 6 at weekly spacing, which blunts the
headline coverage-through-shift chart. Fortnightly keeps roughly 3.

Whatever is chosen, run it **alone**. See the warning below.

### Run it with

    uv run headroom timings --step 28          # confirm the rate on an idle machine first
    uv run headroom stats --models ETS,Theta,MSTL --step 28

It checkpoints to `backtest/out/` every five origins and resumes from where it stopped, so
it is safe to interrupt. A rerun with the same options continues; a rerun with different
options refuses rather than mixing two schedules into one table.

### Warning: do not trust timings taken on a busy machine

The same AutoARIMA fit, same origin, same data, was measured at **166, 215, 310 and 380
seconds** depending on what else was competing for the twelve cores. One of those runs
asked for 99 quantile levels and one for 3, and the 99-level run was the *faster* of the
two, which is how it became clear the variance was contention and not workload.

Two lessons, both already acted on. Timings go in `docs/methods.md` only when the machine
was idle. And `PLAN.md` section 1 promises "compute time per method" as a reported number,
so the real run has to be clean or that number is not a measurement.

---

## What has been found, in order of how much it matters

### 1. Adaptive conformal cannot widen past its calibration window

The headline finding, and it is not what the plan expected. `PLAN.md` section 9 predicted
adaptive conformal would "track nominal within weeks" through the 2020 shift. It does not,
and the reason is structural rather than a tuning failure.

**Driving `alpha` to zero buys the largest nonconformity score in the calibration window
and not one unit more.** Measured at the city through March 2020:

| Step size | Mean alpha in the shift | At the floor | Mean width | Window's widest possible | Worst residual |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0.1000 | 0% | 790 | 1374 | 1574 |
| 0.05 | 0.0409 | 8% | 1262 | 1374 | 1574 |
| 0.20 | 0.1016 | 15% | 970 | 1374 | 1574 |

At `gamma = 0.05` the method reached 1,262 against a hard ceiling of 1,374, and the worst
residual was 1,574. **No value of alpha could have covered that day.** It ran out of room;
it was not too slow.

The standard remedy is a **scale-free nonconformity score**: divide each residual by a
local volatility estimate so the interval can exceed anything the window has literally
seen. That is the obvious next step for the conformal work and it is named in
`docs/methods.md` rather than quietly attempted. Nothing in the repository claims it.

### 2. Conformal fixes the baseline's undercoverage, at a price in width

At 90 percent nominal, city level, 964 weekly origins:

| Method | Coverage | Mean width | Worst 91-day window |
|---|---:|---:|---:|
| Base quantiles | 0.8834 | 885.6 | 0.156 |
| Split conformal | 0.8965 | 982.8 | 0.582 |
| Adaptive, gamma 0.05 | 0.8969 | 1055.4 | **0.670** |
| Aggregated, 6 experts | 0.8941 | 999.6 | 0.610 |

The worst window is the March 2020 number. Conformal lifts it from 0.156 to between 0.58
and 0.67; adaptive at a step size that can actually move beats split conformal by 8.8
points. None of them reaches nominal, for the reason above.

### 3. The training window was expanding, and that was a bug

Fixed 2026-09-12. Each forecast now sees the 1,095 days before its origin. Previously it
saw everything back to 2005, which meant the last origin was fitted on 21 years spanning
two regime changes, and the training length ran from 1,095 days to 7,829, so every fit got
more expensive and a run's cost could not be projected from its first origins. That is
exactly how the first compute estimate came out about half of what it should have been.

The fix improved the result: CRPS unchanged at 163.12 against 163.19 (far inside either
interval), **coverage up from 0.8643 to 0.8834**, width up from 829 to 886, runtime halved.

### 4. Three smaller bugs, each caught by its own test

- **CRPS cannot pad its integrand with zero at the ends.** The integrand only vanishes
  there when the observation lies inside the forecast's support, so padding discards part
  of the penalty exactly when an observation lands outside the grid, which is what a shift
  produces. About half a percent of the score, growing with the miss, and only ever for
  the forecasts that missed. Each tail is now integrated in closed form under an explicit
  flat-tail assumption that is conservative rather than flattering. The check: a point
  forecast now scores exactly the absolute error.
- **The conformal quantile has to be an order statistic**, not a call to a quantile
  function. The usual convention interpolates over `n - 1` intervals and lands one order
  statistic away from `k = ceil((n + 1)(1 - alpha))`. On an exchangeable fixture that
  over-covered at 0.9255 against a 0.90 target. It is invisible in a coverage table,
  because over-covering reads as caution rather than as a bug.
- **The conformal feedback lag is two origins, not one.** With weekly origins, the outcome
  of a 14-day-ahead forecast made at the previous origin has not happened yet.
  `available_upto` is the only place that rule lives, and the test corrupts the future of
  a score series and asserts that no width already produced changes.

### 5. The block bootstrap's block length is not driving the intervals, but ignoring
dependence would

Independent resampling halves the city CRPS interval, 23.9 wide down to 12.1. From 14
origins upward the width is flat to within 3 percent, so 28 is a safe choice rather than a
tuned one. Table in `docs/methods.md`.

---

## Rule C candidates, with the evidence so far

`PLAN.md` section 9 lists three. The evidence now points somewhere the plan did not
anticipate, which is worth more than confirming it would have been.

1. **Split conformal through the shift.** Evidence exists: worst window 0.582 against
   adaptive's 0.670. Real but smaller than the plan expected, because split conformal's
   rolling calibration window recalibrates it within about a year anyway.
2. **Choosing the model by MAE.** Not yet testable; needs the statistical models.
3. **A global neural model as the default.** Week 2.
4. **New: adaptive conformal cannot widen past its calibration window.** Finding 1 above.
   This is the strongest candidate: it is measured, it is structural, it explains a
   negative result the plan expected to be positive, and the remedy is nameable. Written
   up in `docs/methods.md` and in `headroom.conformal.aci`.

---

## Reproducing what exists

    uv sync --extra stats
    uv run headroom data build      # ~1 h first time; cached per year afterwards
    uv run headroom baseline        # ~4 min
    uv run headroom conformal       # ~10 min

`uv run headroom --help` lists everything. Data lives in `data/` and backtest output in
`backtest/out/`; both are gitignored and neither is ever committed.

---

## Watch out for

- **The README tables are filled by the report command, never by hand** (`CLAUDE.md`).
  The command does not exist yet, so the tables are still empty and should stay that way.
- **Every reported number carries a confidence interval.** `headroom.score.bootstrap`.
- **State the conformal assumption wherever coverage is shown.** Adaptive conformal
  guarantees long-run average coverage, not per-period. Saying otherwise is the single
  most likely way for this project to be wrong in public.
- **The repository is public when it goes live.** Nothing from the private plan comes
  across except the one-liner and the technical line.
- **One test failed once and has not failed since.**
  `test_resuming_an_interrupted_run_produces_the_same_forecasts`, with a Windows
  `PermissionError` on the checkpoint's atomic rename. Six consecutive full runs afterwards
  were clean, so it was not reproduced. The cause is almost certainly a file briefly held
  by an indexer or virus scanner, which is transient and clears in milliseconds. The save
  now retries the rename five times with a 0.2 second pause and only then gives up, because
  an unhandled one would end a nine-hour run. If it recurs, that retry is where to look.
- **The refit schedule is documented but not applied.** `docs/methods.md` and
  `headroom.backtest.origins` say models are refitted every fourth origin, and the
  `Origin.refit` flag is computed, but `headroom.backtest.run` never reads it, so every
  model is refitted at every origin. That is the fairer schedule and is what the running
  statistical backtest does. Found 2026-09-13 while the run was in progress and
  deliberately not changed under it. Before the neural models, decide which is true and
  make the code and the docs agree: neural refits at every origin are unlikely to be
  affordable, and the statistical models were meant to be held to the same schedule.
- **The plan repository** (`../ml-portfolio-plan`) has `STATUS.md`, which is edited by
  several sessions at once. Check `git status` there before committing, and commit only
  the files you changed.
