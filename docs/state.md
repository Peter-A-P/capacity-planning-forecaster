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

**181 tests, `ruff` and `mypy --strict` clean.** Run `uv run pytest -q`; add `--run-slow`
for the tests that fit a real model, `--run-network` for the ones that fetch.

| File | Tests | Covers |
|---|---:|---|
| `test_backtest.py` | 38 | Origins and look-ahead, seasonal naive, the runner, checkpointing |
| `test_boosting.py` | 26 | LightGBM rows never read past their anchor, target-day calendar, the fit |
| `test_predictive.py` | 7 | The predictive distribution's feedback rule, ranks, coverage, burn-in |
| `test_score.py` | 34 | CRPS against the closed form, pinball, coverage, width, skill, block bootstrap |
| `test_conformal.py` | 20 | The feedback rule, the conformal quantile, split, adaptive, aggregated |
| `test_data.py` | 20 | The loader, the borough judgement call, checks, the calendar |
| `test_stats_models.py` | 16 | The StatsForecast wrapper and the batched path |
| `test_hierarchy.py` | 14 | The summing matrix and coherence |
| `test_cli.py` | 6 | `HEADROOM_OUT`, and checkpoint names shared by `stats` and `score` |

### Built and measured

| Area | Module | State |
|---|---|---|
| Data | `headroom.data.nyc_ems`, `checks`, `calendar` | Done. 29,977,935 incidents loaded, checked, documented |
| Hierarchy | `headroom.hierarchy` | Done. 37 nodes, coherence error 0.0 |
| Scores | `headroom.score` | Done. CRPS, pinball, coverage, width, skill, block bootstrap |
| Backtest | `headroom.backtest` | Done. 964 weekly origins, resumable checkpointing |
| Baseline | `headroom.models.baselines` | Done. Seasonal naive with per-horizon empirical residual quantiles |
| Conformal | `headroom.conformal` | Done. Split, adaptive (ACI), aggregated (AgACI) |
| Statistical models | `headroom.models.stats` | Done for ETS, Theta, MSTL: 964 weekly origins, scored. AutoARIMA deferred |
| LightGBM | `headroom.models.boosting`, `headroom.conformal.predictive` | Done. 964 weekly origins, scored: ties ETS at every level |
| CLI | `headroom.cli` | Done for what exists, including `boost` and `score` |

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

## Done: LightGBM, global, ties ETS

**Run 2026-09-13 22:33 to 2026-09-14 01:12, scored 2026-09-14.** Tables in
`docs/methods.md` under "LightGBM, global". Paired CRPS difference against ETS on origins
53 to 963 (the first 53 have no calibration window for its distribution):

| Level | LightGBM minus ETS | LightGBM skill against seasonal naive | Coverage at 90% |
|---|---|---|---|
| City | -1.65 [-6.81, +5.98] | +0.190 [+0.149, +0.223] | 0.897 |
| Borough | -0.48 [-1.25, +0.62] | +0.222 [+0.196, +0.244] | 0.899 |
| Dispatch area | +0.007 [-0.067, +0.098] | +0.252 [+0.239, +0.262] | 0.903 |

A global model with holidays bought nothing measurable over ETS. That is the reference the
neural models now have to beat to claim anything for deep learning.

**Compute, corrected.** Fits were a steady 6.5 to 10.8 seconds from 23:45 onward; before
that something else was using the cores. The clean cost is about 7 seconds a fit, about
1.9 hours for a weekly run. The 29 seconds and 7.7 hours recorded on 2026-09-13 were a
busy-machine measurement and are retracted in `docs/methods.md` and `PLAN.md`. **Lesson,
again: run timings with nothing else open, including other Claude Code sessions.**

Commands: `uv run headroom boost --step 7`, then
`uv run headroom score --models ETS,Theta,MSTL,LightGBM`.

## Done: weekly origins for ETS, Theta and MSTL

**Run and scored, 2026-09-13.** 964 weekly origins, 13:20 to 21:07, 7.79 hours wall clock
of which 4.80 hours was fitting and 3.0 hours was checkpoint writes. Full tables, with
intervals, in `docs/methods.md` under "The statistical models". City, 90 percent nominal,
each model's own quantiles with no conformal step:

| Method | CRPS | Skill against seasonal naive | Coverage |
|---|---|---|---|
| Seasonal naive | 163.12 [152.58, 176.08] | 0 | 0.883 [0.870, 0.899] |
| ETS | 133.92 [125.28, 143.53] | +0.179 [+0.164, +0.198] | 0.919 [0.908, 0.933] |
| Theta | 135.89 [127.12, 145.79] | +0.167 [+0.151, +0.187] | 0.912 [0.901, 0.926] |
| MSTL | 148.50 [137.26, 162.80] | +0.090 [+0.039, +0.130] | 0.781 [0.762, 0.799] |

ETS and Theta beat the baseline by 17 to 25 percent at every level. **ETS is the best
statistical model**: paired, it beats Theta by 1.3 to 1.5 percent of CRPS with intervals
excluding zero at every level, so ETS is what later models are paired against. MSTL's
median is 10 to 15 percent worse than ETS's and its intervals are 27 percent narrower,
because StatsForecast adds the seasonal forecast to its quantiles as a fixed shift with no
seasonal uncertainty (read from the source; coverage 0.68 one day ahead, 0.85 at 14).
Ranking by MAE gives the same order as CRPS, so Rule C candidate 2 is not supported by
these three models.

The machine changed on 2026-09-13 (machine B in `docs/methods.md`: i5-10400F, 16 GB), and
it fits three to four times faster than the laptop the estimates below were made on. That
is why weekly was affordable. The estimates and reasoning below are kept as the record of
how the decision was made.

**Next for this area:**

- AutoARIMA at weekly origins, about 17 hours alone on machine B. Fix the checkpoint
  writes first (below), or it will spend hours rewriting its own file.
- Done 2026-09-13: ETS against Theta paired, MSTL's intervals explained, MAE ranking
  checked. Tables in `docs/methods.md`; the report command must reproduce them.

The statistical models have to be back-tested over the record, and that is hours of CPU.
The measured cost on machine A, **idle**, 37 series, 1,095-day window, 12 cores:

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
    uv run headroom stats --models ETS,Theta,MSTL --step 7
    uv run headroom score --models ETS,Theta,MSTL --step 7    # about a minute

It checkpoints to the output directory every five origins and resumes from where it
stopped, so it is safe to interrupt. A rerun with the same options continues; a rerun with
different options refuses rather than mixing two schedules into one table.

**The output directory is `HEADROOM_OUT` if set, else `backtest/out/`.** On Peter's
machine it is set to a folder outside OneDrive, because the repository lives in a synced
folder and three weekly checkpoints are 2.2 GB. The data cache stays in `data/`.

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
2. **Choosing the model by MAE.** Not supported by the statistical models: MAE and CRPS
   rank ETS, Theta, MSTL identically at every level. Open for LightGBM and the neural
   models.
3. **A global neural model as the default.** Week 2.
4. **New: adaptive conformal cannot widen past its calibration window.** Finding 1 above.
   This is the strongest candidate: it is measured, it is structural, it explains a
   negative result the plan expected to be positive, and the remedy is nameable. Written
   up in `docs/methods.md` and in `headroom.conformal.aci`.

---

## Reproducing what exists

    uv sync --extra stats
    uv run headroom data build      # ~1 h first time on machine A, 90 s on machine B
    uv run headroom baseline        # ~4 min
    uv run headroom conformal       # ~10 min
    uv run headroom stats --step 7  # ~8 h on machine B
    uv run headroom score --step 7  # ~1 min

`uv run headroom --help` lists everything. Data lives in `data/` and backtest output in
`HEADROOM_OUT` or `backtest/out/`; both are gitignored and neither is ever committed.

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
- **Checkpoint saves are quadratic in the run's length.** Each save rewrites the whole
  compressed file, which reaches about 740 MB per model at 964 origins, so the weekly run
  spent 3.0 of its 7.79 hours saving. Harmless for correctness. Fix before AutoARIMA by
  writing each batch of origins to its own file. The comment in
  `headroom.backtest.store.open_checkpoint` that calls these files "a few megabytes" is
  from before weekly runs and is wrong at this size.
- **Twelve worker processes need a large Windows paging file.** With the default
  system-managed size, workers failed to start with "the paging file is too small". It is
  now a fixed 32 to 48 GB on machine B.
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
