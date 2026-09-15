"""The `headroom` command line.

Every number in `docs/` and in the README comes from one of these commands, so a reader
can reproduce any of them without reading the source first. The commands are ordered the
way a run goes: fetch the data, establish the baseline, wrap it in conformal intervals,
then the expensive model comparison.

    headroom data build          fetch, aggregate, check, and write the panel
    headroom baseline            seasonal naive over every origin, with intervals
    headroom conformal           split, adaptive and aggregated conformal
    headroom stats               the statistical models (hours; resumable)
    headroom timings             measure cost per origin before committing to a run
    headroom boost               the global LightGBM model (hours; resumable)
    headroom neural              N-HiTS, refitted on a schedule (hours; resumable)
    headroom reconcile           MinT on a model's medians: coherence and the change in CRPS
    headroom decide              staffing from each forecast, priced against an oracle
    headroom score               score finished checkpoints as skill against the baseline
    headroom report              fill the README's results tables (the only thing that may)

Backtest output goes to `backtest/out` unless `HEADROOM_OUT` names another directory. A
full weekly checkpoint is about 740 MB per model, which is worth keeping out of a synced
folder.
"""

import os
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import polars as pl
import typer

from headroom.backtest.origins import TRAIN_WINDOW_DAYS, Origin, Origins
from headroom.backtest.run import Scores, run
from headroom.backtest.store import Checkpoint
from headroom.conformal.aci import AdaptiveConformal
from headroom.conformal.agaci import AggregatedConformal
from headroom.conformal.apply import apply
from headroom.conformal.split import SplitConformal
from headroom.data import checks
from headroom.data.nyc_ems import (
    assign_areas_to_boroughs,
    daily_leaf_counts,
    fetch_daily_counts,
)
from headroom.hierarchy.build import Panel, build
from headroom.models.baselines import SeasonalNaive
from headroom.reconcile.mint import ReconciledMedians
from headroom.score import levels as lv
from headroom.score.bootstrap import confidence_interval, skill_interval
from headroom.score.coverage import empirical_coverage

app = typer.Typer(add_completion=False, help=__doc__)
data_app = typer.Typer(help="Fetch and check the demand data.")
app.add_typer(data_app, name="data")

DATA = Path("data/nyc")
PANEL = DATA / "daily.parquet"

#: Environment variable that relocates backtest output, and where it goes without one.
OUT_VARIABLE = "HEADROOM_OUT"
DEFAULT_OUT = Path("backtest/out")

#: Nominal coverage the single-number tables report, matching PLAN.md section 1.
MAIN_NOMINAL = 0.90

#: Models whose checkpoints hold a median only, and the one-level grid they are stored on.
POINT_MODELS = frozenset({"LightGBM"})
POINT_LEVELS = np.array([0.5])


def output_dir() -> Path:
    """Return where backtest output is written and read.

    Read at call time rather than import time, so a command always sees the environment
    it was started in.

    Returns:
        ``HEADROOM_OUT`` if it is set and not blank, otherwise ``backtest/out``.
    """
    configured = os.environ.get(OUT_VARIABLE, "").strip()
    return Path(configured) if configured else DEFAULT_OUT


def _checkpoint_name(model: str, step: int, origins: Origins) -> str:
    """Name a model's checkpoint file after the schedule that produced it.

    Shared by the command that writes checkpoints and the one that scores them, so the
    two cannot disagree about which file belongs to which run.

    Args:
        model: The model's name.
        step: Days between origins.
        origins: The schedule.

    Returns:
        The file name.
    """
    return f"{model}-step{step}-win{origins.train_window or 0}.npz"


def _panel() -> Panel:
    """Load the panel, or explain how to build it.

    Returns:
        The panel.

    Raises:
        typer.Exit: The panel has not been built yet.
    """
    if not PANEL.exists():
        typer.echo(f"{PANEL} does not exist. Run: headroom data build")
        raise typer.Exit(code=1)
    return build(pl.read_parquet(PANEL))


@data_app.command("build")
def data_build(
    start: Annotated[str, typer.Option(help="First day, ISO format.")] = "2005-01-01",
    end: Annotated[str, typer.Option(help="First day after the range.")] = "2026-07-01",
    refresh: Annotated[bool, typer.Option(help="Refetch cached years.")] = False,
) -> None:
    """Fetch NYC EMS dispatches, aggregate to daily counts, and report the checks.

    Takes about an hour on a first run, because Socrata is asked for one grouped year at
    a time. Completed years are cached, so a second run is seconds.

    Args:
        start: First day, ISO format.
        end: First day after the range, ISO format.
        refresh: Refetch years that are already cached.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    counts = fetch_daily_counts(
        date.fromisoformat(start),
        date.fromisoformat(end),
        Path("data/raw/nyc"),
        refresh=refresh,
    )
    counts.write_parquet(DATA / "counts.parquet")
    typer.echo(f"grouped rows {counts.height:,}, incidents {int(counts['n'].sum()):,}")

    areas = assign_areas_to_boroughs(counts)
    disagreement = checks.borough_disagreement(counts, areas)
    typer.echo(
        f"areas {areas.height}; borough disagreement "
        f"{disagreement.moved_incidents:,}/{disagreement.total_incidents:,} "
        f"= {disagreement.share:.5%}"
    )

    leaf = daily_leaf_counts(counts, areas)
    leaf.write_parquet(PANEL)
    missing = checks.missing_days(leaf)
    typer.echo(f"missing days: {len(missing)}")

    panel = build(leaf)
    typer.echo(
        f"panel: {panel.hierarchy.n_nodes} nodes over {panel.hierarchy.n_leaves} areas, "
        f"{len(panel.days)} days {panel.days[0]} to {panel.days[-1]}"
    )
    typer.echo(
        f"dropped {len(panel.dropped_areas)} areas "
        f"({panel.dropped_share:.4%} of incidents): {', '.join(panel.dropped_areas)}"
    )
    typer.echo(f"coherence error {panel.hierarchy.coherence_error(panel.values):g}")


def _schedule(step: int, window: int | None) -> tuple[Panel, Origins]:
    """Build the panel and an origin schedule.

    Args:
        step: Days between origins.
        window: Trailing training window in days, or 0 for expanding.

    Returns:
        The panel and the schedule.
    """
    panel = _panel()
    origins = Origins(days=panel.days, step=step, train_window=window or None)
    return panel, origins


@app.command()
def baseline(
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Run seasonal naive over every origin and print its scores.

    A few minutes. Everything else in the project is reported as skill against this.

    Args:
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
    """
    panel, origins = _schedule(step, window)
    typer.echo(
        f"{len(origins)} origins, {panel.days[origins.first_index]} to "
        f"{panel.days[list(origins)[-1].index]}, window {origins.train_window or 'expanding'}"
    )
    started = time.perf_counter()
    scores = run(SeasonalNaive(), "Seasonal naive", panel.values, panel.hierarchy, origins)
    typer.echo(f"ran in {time.perf_counter() - started:.0f}s")

    nominal_at = list(scores.nominal).index(MAIN_NOMINAL)
    for level in ("city", "borough", "area"):
        point, low, high = confidence_interval(scores.by_origin(scores.crps, level))
        rows = scores.hierarchy.rows_at(level)
        coverage = empirical_coverage(scores.hits[..., nominal_at][:, rows, :])
        width = float(scores.widths[..., nominal_at][:, rows, :].mean())
        typer.echo(
            f"  {level:8} CRPS {point:8.3f} [{low:7.3f}, {high:7.3f}]  "
            f"coverage90 {coverage:.4f}  width {width:8.2f}"
        )


@app.command()
def conformal(
    alpha: Annotated[float, typer.Option(help="Target miscoverage.")] = 0.10,
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Wrap the seasonal naive median in conformal intervals and compare the methods.

    Prints whole-period coverage and the worst rolling window, which is the number the
    March 2020 shift is judged on.

    Args:
        alpha: Target miscoverage, so nominal coverage is ``1 - alpha``.
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
    """
    panel, origins = _schedule(step, window)
    scores = run(SeasonalNaive(), "Seasonal naive", panel.values, panel.hierarchy, origins)
    origin_days = [panel.days[origin.index] for origin in origins]
    window_origins = max(3, 91 // step)

    for method in (
        SplitConformal(alpha),
        AdaptiveConformal(alpha, gamma=0.01),
        AdaptiveConformal(alpha, gamma=0.05),
        AggregatedConformal(alpha),
    ):
        result = apply(method, scores, panel.values, alpha)
        rows = result.hierarchy.rows_at("city")
        hits = np.asarray(result.hits[:, rows, :], dtype=bool).reshape(len(origins), -1)
        valid = np.asarray(result.valid[:, rows, :], dtype=bool).reshape(len(origins), -1)
        per_origin = np.where(
            valid.sum(1) > 0, (hits & valid).sum(1) / np.maximum(valid.sum(1), 1), np.nan
        )
        usable = np.flatnonzero(np.isfinite(per_origin))
        rolling = np.convolve(
            per_origin[usable], np.ones(window_origins) / window_origins, mode="valid"
        )
        worst = int(np.argmin(rolling))
        typer.echo(
            f"  {method.name[:46]:46} coverage {result.coverage('city'):.4f} "
            f"width {result.mean_width('city'):8.2f} "
            f"worst {rolling[worst]:.3f} "
            f"ending {origin_days[usable[worst + window_origins - 1]]}"
        )


@app.command()
def timings(
    step: Annotated[int, typer.Option(help="Days between origins.")] = 14,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Measure what each statistical model costs per origin, before committing to a run.

    Run this on an otherwise idle machine. Timings taken while anything else is running
    are worthless: the same AutoARIMA fit has been measured at 166, 215, 310 and 380
    seconds on this laptop depending on what else was competing for the twelve cores.

    Args:
        step: Days between origins, to project a total.
        window: Trailing training window in days, or 0 to let it expand.
    """
    warnings.filterwarnings("ignore")
    from headroom.models.stats import BatchedModels, catalogue

    panel, origins = _schedule(step, window)
    sampled = list(origins)
    picks = [sampled[i] for i in (len(sampled) // 10, len(sampled) // 2, -2)]

    for model in catalogue():
        batch = BatchedModels((model,), n_jobs=-1)
        taken = []
        for origin in picks:
            started = time.perf_counter()
            batch.forecast_all(panel.values[:, origin.train], origins.horizon, lv.SCORING)
            taken.append(time.perf_counter() - started)
        mean = float(np.mean(taken))
        typer.echo(
            f"  {model.name:10} {mean:7.1f}s/origin  "
            f"({min(taken):.0f} to {max(taken):.0f})  "
            f"{len(origins)} origins = {mean * len(origins) / 3600:.1f}h"
        )


@app.command()
def stats(
    models: Annotated[
        str, typer.Option(help="Comma-separated model names.")
    ] = "ETS,Theta,MSTL",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 28,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    save_every: Annotated[int, typer.Option(help="Origins between checkpoint writes.")] = 5,
) -> None:
    """Back-test the statistical models, checkpointing so it can be resumed.

    Hours, not minutes. Run `headroom timings` first and run this on an idle machine.
    Stopping it is safe: every few origins are written to the output directory (see
    `HEADROOM_OUT`) and a rerun with the same options picks up where it left off.

    Args:
        models: Comma-separated names from the catalogue.
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
        save_every: Origins between checkpoint writes.
    """
    warnings.filterwarnings("ignore")
    from headroom.backtest.store import open_checkpoint
    from headroom.models.stats import BatchedModels, catalogue

    wanted = [name.strip() for name in models.split(",") if name.strip()]
    known = {model.name: model for model in catalogue()}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        typer.echo(f"unknown models {unknown}; choose from {sorted(known)}")
        raise typer.Exit(code=1)

    panel, origins = _schedule(step, window)
    batch = BatchedModels(tuple(known[name] for name in wanted), n_jobs=-1)
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    points = {
        name: open_checkpoint(
            out / _checkpoint_name(name, step, origins),
            name,
            origins,
            panel.hierarchy.n_nodes,
            lv.SCORING,
        )
        for name in wanted
    }

    typer.echo(f"{len(origins)} origins at step {step}, models {batch.names}, output {out}")
    typer.echo(
        f"resuming from origin {min(p.n_done for p in points.values())} of {len(origins)}"
    )

    started = time.perf_counter()
    computed = 0
    for origin in origins:
        if all(point.done[origin.number] for point in points.values()):
            continue
        began = time.perf_counter()
        produced = batch.forecast_all(
            panel.values[:, origin.train], origins.horizon, lv.SCORING
        )
        elapsed = time.perf_counter() - began
        for name, quantiles in produced.items():
            points[name].record(origin.number, quantiles, elapsed / len(produced))
        computed += 1
        if computed % save_every == 0:
            for point in points.values():
                point.save()
            done = min(point.n_done for point in points.values())
            rate = (time.perf_counter() - started) / computed
            typer.echo(
                f"  origin {origin.number + 1}/{len(origins)} ({panel.days[origin.index]}) "
                f"{elapsed:.0f}s  mean {rate:.0f}s  "
                f"~{(len(origins) - done) * rate / 3600:.1f}h left"
            )
    for point in points.values():
        point.save()
    typer.echo(f"done in {(time.perf_counter() - started) / 3600:.2f}h")


@app.command()
def boost(
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    save_every: Annotated[int, typer.Option(help="Origins between checkpoint writes.")] = 10,
) -> None:
    """Back-test the global LightGBM model, refitting at every origin, resumably.

    About two hours: roughly 7 seconds a fit on an idle six-core machine (`docs/methods.md`).
    Run on an idle machine; the total is reported as the model's compute. The checkpoint
    holds the median forecast only; `headroom score` builds the distribution from its past
    errors.

    Args:
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
        save_every: Origins between checkpoint writes.
    """
    from headroom.backtest.store import open_checkpoint
    from headroom.models.boosting import GlobalLightGBM

    panel, origins = _schedule(step, window)
    model = GlobalLightGBM(levels=panel.hierarchy.levels)
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = open_checkpoint(
        out / _checkpoint_name(model.name, step, origins),
        model.name,
        origins,
        panel.hierarchy.n_nodes,
        POINT_LEVELS,
    )
    typer.echo(f"{len(origins)} origins at step {step}, {model.name}, output {out}")
    typer.echo(f"resuming from origin {checkpoint.n_done} of {len(origins)}")

    started = time.perf_counter()
    computed = 0
    for origin in origins:
        if checkpoint.done[origin.number]:
            continue
        # The model gets demand up to the origin and dates only, never a later value.
        days = panel.days[origin.train.start : origin.index + 1 + origins.horizon]
        began = time.perf_counter()
        median = model.forecast(panel.values[:, origin.train], origins.horizon, days)
        elapsed = time.perf_counter() - began
        checkpoint.record(origin.number, median[..., np.newaxis], elapsed)
        computed += 1
        if computed % save_every == 0:
            checkpoint.save()
            rate = (time.perf_counter() - started) / computed
            left = (len(origins) - checkpoint.n_done) * rate / 3600
            typer.echo(
                f"  origin {origin.number + 1}/{len(origins)} ({origin.day}) {elapsed:.0f}s  "
                f"mean {rate:.0f}s  ~{left:.1f}h left"
            )
    checkpoint.save()
    typer.echo(f"done in {(time.perf_counter() - started) / 3600:.2f}h")


@app.command()
def neural(
    refit_every: Annotated[
        int, typer.Option(help="Origins between refits; forecasts are made at every origin.")
    ] = 13,
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    save_every: Annotated[int, typer.Option(help="Origins between checkpoint writes.")] = 13,
) -> None:
    """Back-test N-HiTS, refitting on a schedule and forecasting every origin, resumably.

    A fit is minutes on CPU and a forecast from fitted weights is a fraction of a second,
    so the model is refitted every ``refit_every`` origins and forecasts each origin in
    between from its own fresh inputs. The checkpoint is named for the schedule, so
    results from two schedules cannot be mixed. Each origin's recorded time is its share
    of the fit it used plus its own forecast.

    Args:
        refit_every: Origins between refits.
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
        save_every: Origins between checkpoint writes.
    """
    import logging

    from headroom.backtest.store import open_checkpoint
    from headroom.models.neural import NHiTS

    warnings.filterwarnings("ignore")
    for noisy in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    panel = _panel()
    origins = Origins(
        days=panel.days, step=step, train_window=window or None, refit_every=refit_every
    )
    model = NHiTS(levels=lv.SCORING, horizon=origins.horizon)
    label = neural_label(model.name, refit_every)
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = open_checkpoint(
        out / _checkpoint_name(label, step, origins),
        label,
        origins,
        panel.hierarchy.n_nodes,
        lv.SCORING,
    )
    listed = list(origins)
    typer.echo(
        f"{len(origins)} origins at step {step}, {label}: {origins.n_refits} fits, output {out}"
    )
    typer.echo(f"resuming from origin {checkpoint.n_done} of {len(origins)}")

    started = time.perf_counter()
    fitted_for: int | None = None
    fit_seconds = 0.0
    computed = 0
    for origin in origins:
        if checkpoint.done[origin.number]:
            continue
        refit_at = origins.refit_for(origin.number)
        if fitted_for != refit_at:
            began = time.perf_counter()
            model.fit(panel.values[:, listed[refit_at].train])
            fit_seconds = time.perf_counter() - began
            fitted_for = refit_at
            typer.echo(
                f"  fit at origin {refit_at} ({listed[refit_at].day}) {fit_seconds:.0f}s"
            )
        began = time.perf_counter()
        quantiles = model.predict(panel.values[:, origin.train])
        share = fit_seconds / refit_every
        checkpoint.record(origin.number, quantiles, share + time.perf_counter() - began)
        computed += 1
        if computed % save_every == 0:
            checkpoint.save()
            rate = (time.perf_counter() - started) / computed
            left = (len(origins) - checkpoint.n_done) * rate / 3600
            typer.echo(
                f"  origin {origin.number + 1}/{len(origins)} ({origin.day})  "
                f"mean {rate:.0f}s  ~{left:.1f}h left"
            )
    checkpoint.save()
    typer.echo(f"done in {(time.perf_counter() - started) / 3600:.2f}h")


def neural_label(name: str, refit_every: int) -> str:
    """Name a neural run after its refit schedule, for its checkpoint and its tables.

    Args:
        name: The model's name.
        refit_every: Origins between refits.

    Returns:
        For example ``"N-HiTS-refit13"``.
    """
    return f"{name}-refit{refit_every}"


@app.command()
def reconcile(
    model: Annotated[str, typer.Option(help="A model with a quantile checkpoint.")] = "ETS",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Reconcile a model's medians with MinT and report coherence and the change in CRPS.

    ``W`` comes from the model's own errors at the same horizon step over the previous 52
    origins, so the first 53 origins are not reconciled and every number here is on
    origins 53 onward. Each node's quantiles move with its median.

    Args:
        model: The model whose finished checkpoint is reconciled.
        step: Days between origins the checkpoint was built with.
        window: Trailing training window it was built with, or 0 for expanding.

    Raises:
        typer.Exit: The checkpoint is missing, incomplete, or holds a median only.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.backtest.store import open_checkpoint
    from headroom.reconcile.mint import reconcile_backtest, shift_quantiles

    if model in POINT_MODELS:
        typer.echo(f"{model} stores a median only; reconcile a quantile model")
        raise typer.Exit(code=1)
    panel, origins = _schedule(step, window)
    path = output_dir() / _checkpoint_name(model, step, origins)
    if not path.exists():
        typer.echo(f"no checkpoint at {path}")
        raise typer.Exit(code=1)
    checkpoint = open_checkpoint(path, model, origins, panel.hierarchy.n_nodes, lv.SCORING)
    if not checkpoint.complete:
        typer.echo(f"{model}: {checkpoint.n_done} of {len(origins)} origins done")
        raise typer.Exit(code=1)

    actual = np.stack([panel.values[:, origin.target] for origin in origins])
    median_at = int(np.abs(lv.SCORING - 0.5).argmin())
    base_median = checkpoint.quantiles[..., median_at]
    started = time.perf_counter()
    result = reconcile_backtest(base_median, actual, panel.hierarchy, step)
    took = time.perf_counter() - started
    start = result.first_valid

    base_breach = max(
        panel.hierarchy.coherence_error(base_median[o, :, h])
        for o in range(start, len(origins))
        for h in range(origins.horizon)
    )
    intensity = result.intensity[start:]
    typer.echo(
        f"{model}, origins {start} to {len(origins) - 1}, reconciled in {took:.0f}s\n"
        f"  coherence error, largest breach in incidents: base {base_breach:.1f}, "
        f"reconciled {result.coherence_error:.2e}\n"
        f"  shrinkage intensity: median {np.median(intensity):.3f}, "
        f"range {intensity.min():.3f} to {intensity.max():.3f}"
    )

    reconciled_quantiles = shift_quantiles(
        checkpoint.quantiles[start:], base_median[start:], result.medians[start:]
    )
    seconds = float(checkpoint.seconds.sum())
    base = score_forecasts(
        model,
        panel.hierarchy,
        origins,
        checkpoint.quantiles[start:],
        actual[start:],
        lv.SCORING,
        seconds,
    )
    del checkpoint
    reconciled = score_forecasts(
        f"{model} + MinT",
        panel.hierarchy,
        origins,
        reconciled_quantiles,
        actual[start:],
        lv.SCORING,
        seconds,
    )
    del reconciled_quantiles
    _print_scores(base, None, None)
    _print_scores(reconciled, None, base)


@app.command()
def decide(
    models: Annotated[
        str, typer.Option(help="Comma-separated models with finished checkpoints.")
    ] = "ETS,LightGBM",
    reconciled: Annotated[
        str, typer.Option(help="Quantile models also staffed from their MinT reconciliation.")
    ] = "ETS",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Staff every dispatch area from each forecast and price it against an oracle.

    Inputs come from `inputs/decision.toml` and are illustrative. Each area is staffed for
    each of the fourteen days ahead at the service level the costs imply and at the fixed
    levels in the table; the oracle staffs what each day actually needed and costs
    nothing. Reported per method: the realised cost per day summed over the areas, the
    units staffed per day, and the paired difference in cost against the first model
    listed. Every method is on origins 53 onward, where MinT and LightGBM's distribution
    exist, so all rows are paired.

    Args:
        models: Models whose finished checkpoints are staffed from; the first is the
            reference for paired differences. Seasonal naive is always added.
        reconciled: Quantile models also staffed after MinT reconciliation.
        step: Days between origins the checkpoints were built with.
        window: Trailing training window they were built with, or 0 for expanding.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete.
    """
    from headroom.conformal.predictive import first_valid_origin
    from headroom.decide.newsvendor import critical_ratio, load_inputs

    inputs = load_inputs()
    implied = critical_ratio(inputs.cost_over, inputs.cost_under)
    service = _service_levels(implied, inputs.service_levels)
    wanted = [name.strip() for name in models.split(",") if name.strip()]
    to_reconcile = {name.strip() for name in reconciled.split(",") if name.strip()}

    panel, origins = _schedule(step, window)
    start = first_valid_origin(origins.horizon, step)
    listed = list(origins)
    actual = np.stack([panel.values[:, origin.target] for origin in origins])
    typer.echo(
        f"{len(panel.hierarchy.rows_at('area'))} dispatch areas, origins {start} to "
        f"{len(origins) - 1} ({listed[start].day} to {listed[-1].day})\n"
        f"inputs (illustrative): {inputs.demand_per_unit:g} incidents per unit, "
        f"cost {inputs.cost_over:g} per spare unit-day and {inputs.cost_under:g} per missing "
        f"one, so the costs imply staffing at the {implied:.0%} quantile"
    )

    demand = {"Seasonal naive": _naive_demand(panel, origins, start, service)}
    for name in wanted:
        checkpoint = _complete_checkpoint(name, step, origins, panel.hierarchy.n_nodes)
        quantiles = _distribution(name, checkpoint.quantiles, actual, step)
        demand[name] = _area_demand(quantiles, panel, start, service)
        if name in to_reconcile and name not in POINT_MODELS:
            shifted, _ = _mint_quantiles(quantiles, actual, panel, step)
            demand[f"{name} + MinT"] = _area_demand(shifted, panel, start, service)
            del shifted
        del checkpoint, quantiles

    reference = wanted[0] if wanted else "Seasonal naive"
    staffing = _staffing(demand, actual[start:], panel, service)
    typer.echo(f"oracle: {staffing.oracle_units:.1f} units a day across the areas, cost 0")
    for j, level in enumerate(service):
        mark = " (implied by the costs)" if abs(level - implied) < 1e-9 else ""
        typer.echo(f"service level {level:.0%}{mark}")
        for name in demand:
            daily = staffing.costs[name][j]
            cost, low, high = confidence_interval(daily)
            line = (
                f"  {name:16} cost/day {cost:7.2f} [{low:7.2f}, {high:7.2f}]  "
                f"units/day {staffing.units[name][j].mean():6.1f}"
            )
            if name != reference and reference in staffing.costs:
                d, d_low, d_high = confidence_interval(daily - staffing.costs[reference][j])
                line += f"  minus {reference} {d:+6.2f} [{d_low:+6.2f}, {d_high:+6.2f}]"
            typer.echo(line)


def _service_levels(implied: float, fixed: tuple[float, ...]) -> list[float]:
    """Return the service levels staffed at: the one the costs imply, then the fixed ones.

    Args:
        implied: The critical ratio the costs imply.
        fixed: The levels in the decision table.

    Returns:
        The distinct levels, ascending.
    """
    return sorted({round(implied, 9), *fixed})


def _complete_checkpoint(name: str, step: int, origins: Origins, n_nodes: int) -> Checkpoint:
    """Open a model's finished checkpoint, or explain why it cannot be used.

    Args:
        name: The model's checkpoint name.
        step: Days between origins it was built with.
        origins: The schedule it was built with.
        n_nodes: Nodes in the hierarchy.

    Returns:
        The checkpoint, every origin done.

    Raises:
        typer.Exit: The checkpoint is missing or incomplete.
    """
    from headroom.backtest.store import open_checkpoint

    path = output_dir() / _checkpoint_name(name, step, origins)
    if not path.exists():
        typer.echo(f"no checkpoint at {path}; run the model first, or set {OUT_VARIABLE}")
        raise typer.Exit(code=1)
    levels = POINT_LEVELS if name in POINT_MODELS else lv.SCORING
    checkpoint = open_checkpoint(path, name, origins, n_nodes, levels)
    if not checkpoint.complete:
        typer.echo(f"{name}: {checkpoint.n_done} of {len(origins)} origins done")
        raise typer.Exit(code=1)
    return checkpoint


def _distribution(
    name: str,
    quantiles: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    step: int,
) -> npt.NDArray[np.float64]:
    """Return a model's forecast distribution on the scoring grid over every origin.

    A median-only model is given the conformal predictive distribution from its own past
    errors, which is ``nan`` before the first origin with a full calibration window.

    Args:
        name: The model's name.
        quantiles: Its checkpoint's forecasts.
        actual: What happened, shape ``(n_origins, n_nodes, horizon)``.
        step: Days between origins.

    Returns:
        Quantiles, shape ``(n_origins, n_nodes, horizon, n_scoring)``.
    """
    from headroom.conformal.predictive import predictive_quantiles

    if name not in POINT_MODELS:
        return quantiles
    return predictive_quantiles(quantiles[..., 0], actual, lv.SCORING, step).quantiles


def _mint_quantiles(
    quantiles: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    panel: Panel,
    step: int,
) -> tuple[npt.NDArray[np.float64], ReconciledMedians]:
    """Reconcile a model's medians with MinT and move each node's quantiles with them.

    Args:
        quantiles: The model's forecasts over every origin.
        actual: What happened.
        panel: The panel, for its hierarchy.
        step: Days between origins.

    Returns:
        The shifted quantiles, ``nan`` before the first reconciled origin, and the
        reconciliation itself.
    """
    from headroom.reconcile.mint import reconcile_backtest, shift_quantiles

    median_at = int(np.abs(lv.SCORING - 0.5).argmin())
    base_median = quantiles[..., median_at]
    mint = reconcile_backtest(base_median, actual, panel.hierarchy, step)
    start = mint.first_valid
    shifted = np.full_like(quantiles, np.nan)
    shifted[start:] = shift_quantiles(
        quantiles[start:], base_median[start:], mint.medians[start:]
    )
    return shifted, mint


def _area_demand(
    quantiles: npt.NDArray[np.float64], panel: Panel, start: int, service: list[float]
) -> npt.NDArray[np.float64]:
    """Read each dispatch area's demand at each service level, from ``start`` onward.

    Args:
        quantiles: Forecasts on the scoring grid over every origin.
        panel: The panel, for its hierarchy.
        start: First origin staffed.
        service: Service levels.

    Returns:
        Demand, shape ``(n_origins - start, n_areas, horizon, n_service)``.
    """
    from headroom.decide.newsvendor import demand_at

    area_q = quantiles[start:, panel.hierarchy.rows_at("area"), :, :]
    return np.stack([demand_at(area_q, lv.SCORING, level) for level in service], axis=-1)


def _naive_demand(
    panel: Panel, origins: Origins, start: int, service: list[float]
) -> npt.NDArray[np.float64]:
    """Seasonal naive's area demand at each service level, from ``start`` onward.

    Args:
        panel: The panel.
        origins: The schedule.
        start: First origin staffed.
        service: Service levels.

    Returns:
        Demand, shaped as :func:`_area_demand` returns it.
    """
    areas = panel.hierarchy.rows_at("area")
    grid = np.array(service)
    naive = SeasonalNaive()
    return np.stack(
        [
            naive.forecast(panel.values[:, origin.train], origins.horizon, grid)[areas]
            for origin in list(origins)[start:]
        ]
    )


@dataclass(frozen=True, slots=True)
class Staffing:
    """Realised staffing per method and service level.

    Attributes:
        costs: Per method, one array per service level of cost per day summed over the
            areas, one value per origin.
        units: Per method, the same shape, of units staffed per day summed over the areas.
        oracle_units: Units a day the oracle staffs, knowing demand, averaged over origins.
    """

    costs: dict[str, list[npt.NDArray[np.float64]]]
    units: dict[str, list[npt.NDArray[np.float64]]]
    oracle_units: float


def _staffing(
    demand: dict[str, npt.NDArray[np.float64]],
    actual: npt.NDArray[np.float64],
    panel: Panel,
    service: list[float],
) -> Staffing:
    """Staff from each method's demand and price it against what happened.

    Args:
        demand: Per method, area demand as :func:`_area_demand` returns it.
        actual: What happened over the same origins, every node.
        panel: The panel, for its hierarchy.
        service: Service levels, matching the last axis of each demand array.

    Returns:
        Costs and units per method and service level.
    """
    from headroom.decide.newsvendor import load_inputs, realised_cost, units_for

    inputs = load_inputs()
    area_actual = actual[:, panel.hierarchy.rows_at("area"), :]
    costs: dict[str, list[npt.NDArray[np.float64]]] = {}
    units: dict[str, list[npt.NDArray[np.float64]]] = {}
    for name, forecast in demand.items():
        costs[name], units[name] = [], []
        for j, _ in enumerate(service):
            staffed = units_for(forecast[..., j], inputs.demand_per_unit)
            costs[name].append(
                realised_cost(staffed, area_actual, inputs).sum(axis=1).mean(axis=1)
            )
            units[name].append(staffed.sum(axis=1).mean(axis=1))
    oracle = float(units_for(area_actual, inputs.demand_per_unit).sum(axis=1).mean())
    return Staffing(costs=costs, units=units, oracle_units=oracle)


@app.command()
def score(
    models: Annotated[
        str, typer.Option(help="Comma-separated model names.")
    ] = "ETS,Theta,MSTL",
    against: Annotated[
        str, typer.Option(help="Model every other is paired against; blank for none.")
    ] = "ETS",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Score finished checkpoints as skill against seasonal naive, and paired.

    Seasonal naive is rerun on the same origins, so every comparison is paired. Each
    number carries a 95 percent block-bootstrap interval.

    Coverage for the statistical models is of their own prediction quantiles, with no
    conformal step, so no coverage guarantee is claimed. A model that forecasts only a
    median (LightGBM) is given a conformal predictive distribution from its own past
    errors (`headroom.conformal.predictive`), which exists only from the first origin
    with a full calibration window. When one is included, **every model is scored on the
    origins from that one onward**, so the tables stay paired and the numbers differ from
    a run without it.

    Args:
        models: Comma-separated names of models whose checkpoints are complete.
        against: The model each other model's CRPS is differenced against, per origin.
        step: Days between origins the checkpoints were built with.
        window: Trailing training window they were built with, or 0 for expanding.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.backtest.store import open_checkpoint
    from headroom.conformal.predictive import first_valid_origin, predictive_quantiles

    wanted = [name.strip() for name in models.split(",") if name.strip()]
    panel, origins = _schedule(step, window)
    out = output_dir()

    paths = {name: out / _checkpoint_name(name, step, origins) for name in wanted}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        typer.echo(
            f"no checkpoint at {missing}; run headroom stats or boost first, or set "
            f"{OUT_VARIABLE}"
        )
        raise typer.Exit(code=1)

    point_models = [name for name in wanted if name in POINT_MODELS]
    start = first_valid_origin(origins.horizon, step) if point_models else 0
    listed = list(origins)
    typer.echo(
        f"origins {start} to {len(origins) - 1} of {len(origins)} at step {step} "
        f"({listed[start].day} to {listed[-1].day}), output {out}"
    )
    if point_models:
        typer.echo(
            f"{', '.join(point_models)}: median only, distribution from its own errors over "
            "the previous 52 origins; the only model given holiday features"
        )

    actual = np.stack([panel.values[:, origin.target] for origin in origins])
    baseline_scores = _from_origin(
        run(SeasonalNaive(), "Seasonal naive", panel.values, panel.hierarchy, origins), start
    )
    everything: dict[str, Scores] = {}
    for name in wanted:
        levels = POINT_LEVELS if name in POINT_MODELS else lv.SCORING
        checkpoint = open_checkpoint(
            paths[name], name, origins, panel.hierarchy.n_nodes, levels
        )
        if not checkpoint.complete:
            typer.echo(
                f"{name}: {checkpoint.n_done} of {len(origins)} origins done; not scored"
            )
            raise typer.Exit(code=1)
        if name in POINT_MODELS:
            quantiles = predictive_quantiles(
                checkpoint.quantiles[..., 0], actual, lv.SCORING, step
            ).quantiles
        else:
            quantiles = checkpoint.quantiles
        everything[name] = score_forecasts(
            name=name,
            hierarchy=panel.hierarchy,
            origins=origins,
            quantiles=quantiles[start:],
            actual=actual[start:],
            scoring=lv.SCORING,
            fit_seconds=float(checkpoint.seconds.sum()),
        )
        del checkpoint, quantiles

    reference = everything.get(against.strip()) if against.strip() else None
    _print_scores(baseline_scores, None, None)
    for scores in everything.values():
        paired = reference if reference is not None and reference is not scores else None
        _print_scores(scores, baseline_scores, paired)


#: The file `headroom report` writes its tables into.
README = Path("README.md")

#: Hierarchy levels in table order, with the names the tables use.
LEVEL_NAMES = {"city": "City", "borough": "Borough", "area": "Dispatch area"}

#: Days in the rolling window the worst-coverage column is measured over.
SHIFT_WINDOW_DAYS = 91


@app.command()
def report(
    statistical: Annotated[
        str, typer.Option(help="Statistical models; the best by mean CRPS skill is shown.")
    ] = "ETS,Theta,MSTL",
    boosting: Annotated[
        str, typer.Option(help="The gradient-boosting checkpoint.")
    ] = "LightGBM",
    neural: Annotated[
        str, typer.Option(help="Comma-separated neural checkpoints.")
    ] = "N-HiTS-refit4",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    write: Annotated[
        bool, typer.Option(help="Write the tables into README.md as well as printing them.")
    ] = True,
) -> None:
    """Fill the README's results tables from the finished checkpoints.

    The only thing allowed to write those tables. Every model is scored on the same
    origins, from the first one where LightGBM's conformal distribution and the MinT
    covariance both exist, so every comparison in the tables is paired. Takes several
    minutes and a few gigabytes of memory, one checkpoint at a time.

    Args:
        statistical: Statistical models to choose the best from, by CRPS skill averaged
            over the three levels. The best is also reconciled and staffed from.
        boosting: The gradient-boosting model's checkpoint name.
        neural: Neural checkpoint names, such as ``N-HiTS-refit4``.
        step: Days between origins the checkpoints were built with.
        window: Trailing training window they were built with, or 0 for expanding.
        write: Write the tables into README.md between its report markers.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete, or the README has no markers.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.conformal.predictive import first_valid_origin
    from headroom.decide.newsvendor import critical_ratio, load_inputs
    from headroom.report import tables

    panel, origins = _schedule(step, window)
    hierarchy = panel.hierarchy
    start = first_valid_origin(origins.horizon, step)
    listed = list(origins)
    actual = np.stack([panel.values[:, origin.target] for origin in origins])
    inputs = load_inputs()
    implied = critical_ratio(inputs.cost_over, inputs.cost_under)
    service = _service_levels(implied, inputs.service_levels)

    def scored(name: str, quantiles: npt.NDArray[np.float64]) -> Scores:
        typer.echo(f"scoring {name}")
        return score_forecasts(
            name, hierarchy, origins, quantiles[start:], actual[start:], lv.SCORING, 0.0
        )

    def compute(checkpoint: Checkpoint) -> float:
        # The median, not the total: an origin slowed by other work or by the machine
        # sleeping through a fit should not count as the model's cost.
        return float(np.median(checkpoint.seconds)) * len(origins)

    typer.echo("running seasonal naive")
    naive_full = run(SeasonalNaive(), "Seasonal naive", panel.values, hierarchy, origins)
    naive = _from_origin(naive_full, start)

    def mean_skill(scores: Scores) -> float:
        return float(
            np.mean(
                [
                    1.0
                    - scores.by_origin(scores.crps, level).mean()
                    / naive.by_origin(naive.crps, level).mean()
                    for level in LEVEL_NAMES
                ]
            )
        )

    candidates: dict[str, Scores] = {}
    seconds: dict[str, float] = {"Seasonal naive": naive_full.fit_seconds}
    for name in _names(statistical):
        checkpoint = _complete_checkpoint(name, step, origins, hierarchy.n_nodes)
        candidates[name] = scored(name, checkpoint.quantiles)
        seconds[name] = compute(checkpoint)
        del checkpoint
    if not candidates:
        typer.echo("name at least one statistical model")
        raise typer.Exit(code=1)
    best = max(candidates, key=lambda name: mean_skill(candidates[name]))
    skills = ", ".join(f"{name} {mean_skill(s):+.4f}" for name, s in candidates.items())
    typer.echo(f"best statistical model: {best} (mean CRPS skill {skills})")

    demand = {"Seasonal naive": _naive_demand(panel, origins, start, service)}
    checkpoint = _complete_checkpoint(best, step, origins, hierarchy.n_nodes)
    demand[best] = _area_demand(checkpoint.quantiles, panel, start, service)
    typer.echo(f"reconciling {best}")
    shifted, mint = _mint_quantiles(checkpoint.quantiles, actual, panel, step)
    if mint.first_valid > start:
        typer.echo(f"MinT starts at origin {mint.first_valid}, after {start}")
        raise typer.Exit(code=1)
    median_at = int(np.abs(lv.SCORING - 0.5).argmin())
    base_breach = max(
        hierarchy.coherence_error(checkpoint.quantiles[o, :, h, median_at])
        for o in range(start, len(origins))
        for h in range(origins.horizon)
    )
    del checkpoint
    reconciled = scored(f"{best} + MinT", shifted)
    demand[f"{best} + MinT"] = _area_demand(shifted, panel, start, service)
    del shifted

    others: dict[str, Scores] = {}
    for name in [*_names(boosting), *_names(neural)]:
        checkpoint = _complete_checkpoint(name, step, origins, hierarchy.n_nodes)
        quantiles = _distribution(name, checkpoint.quantiles, actual, step)
        others[name] = scored(name, quantiles)
        seconds[name] = compute(checkpoint)
        demand[name] = _area_demand(quantiles, panel, start, service)
        del checkpoint, quantiles

    typer.echo("staffing")
    staffing = _staffing(demand, actual[start:], panel, service)
    window_origins = max(3, SHIFT_WINDOW_DAYS // step)

    methods: list[tuple[str, Scores | None, float]] = [
        ("Seasonal naive", naive, seconds["Seasonal naive"]),
        (f"Best statistical: {best}", candidates[best], seconds[best]),
    ]
    methods += [(_method_label(name), others[name], seconds[name]) for name in others]
    methods += [("PatchTST", None, 0.0), ("TimesFM, zero-shot (clean window only)", None, 0.0)]

    skill_rows: list[list[str]] = []
    for label, scores, cost in methods:
        if scores is None:
            skill_rows.append([label, "not built", "", "", "", "", "", ""])
            continue
        skill_rows += _skill_rows(label, scores, naive, cost, window_origins, listed, start)

    reconcile_rows = []
    for level, level_name in LEVEL_NAMES.items():
        before = candidates[best].by_origin(candidates[best].crps, level)
        after = reconciled.by_origin(reconciled.crps, level)
        reconcile_rows.append(
            [
                level_name,
                tables.number(tables.Estimate.of(confidence_interval(before))),
                tables.number(tables.Estimate.of(confidence_interval(after))),
                tables.number(
                    tables.Estimate.of(confidence_interval(after - before)),
                    places=min(3, tables.decimals_for(float(before.mean())) + 1),
                    signed=True,
                ),
                tables.number(
                    tables.Estimate.of(skill_interval(after, before)), places=3, signed=True
                ),
            ]
        )

    staffing_rows = []
    for j, service_level in enumerate(service):
        mark = ", implied by the costs" if abs(service_level - implied) < 1e-9 else ""
        for i, name in enumerate(demand):
            daily = staffing.costs[name][j]
            difference = (
                "reference"
                if name == best
                else tables.number(
                    tables.Estimate.of(confidence_interval(daily - staffing.costs[best][j])),
                    places=2,
                    signed=True,
                )
            )
            staffing_rows.append(
                [
                    f"{service_level:.0%}{mark}" if i == 0 else "",
                    _method_label(name),
                    tables.number(
                        tables.Estimate.of(confidence_interval(staffing.units[name][j])),
                        places=1,
                    ),
                    tables.number(tables.Estimate.of(confidence_interval(daily)), places=2),
                    difference,
                ]
            )

    first, last = listed[start].day, listed[-1].day
    block = "\n".join(
        [
            f"All numbers from `headroom report`: {len(origins) - start} weekly forecast "
            f"origins, {first} to {last}, each forecasting the next {origins.horizon} days "
            f"for {hierarchy.n_nodes} series (the city, {len(hierarchy.rows_at('borough'))} "
            f"boroughs and {len(hierarchy.rows_at('area'))} dispatch areas). Every model is "
            "scored on the same origins, and every interval is a 95 percent moving-block "
            "bootstrap interval over them, paired wherever two methods are compared.",
            "",
            "**Skill against seasonal naive, per hierarchy level**",
            "",
            tables.markdown_table(
                [
                    "Method",
                    "Level",
                    "CRPS skill",
                    "Pinball skill (median)",
                    "Coverage at 90% nominal, whole period",
                    f"Coverage at 90%, worst {SHIFT_WINDOW_DAYS}-day window (last origin)",
                    "Mean width at 90%",
                    "Fit time, full schedule",
                ],
                skill_rows,
            ),
            "",
            "Skill is one minus the method's mean score over seasonal naive's, so higher is "
            "better and zero is no better than the baseline.",
            "",
            "**Coverage, and what it does and does not promise.** Seasonal naive (quantiles "
            "of its own past errors), the statistical models and N-HiTS are scored on their "
            "own quantiles with no conformal step, so no coverage is guaranteed for them. "
            "LightGBM forecasts a median only, and its distribution "
            "is conformal, built from its own errors over the previous 52 origins. Conformal "
            "coverage holds on average over time and only if errors are exchangeable, which "
            "demand through a shift is not, so it is not promised in any one window. The "
            f"worst window is the lowest coverage in a trailing {SHIFT_WINDOW_DAYS}-day "
            f"window ({window_origins} weekly origins), dated by the last origin in it. It is "
            "the single worst stretch of one history, so it carries no bootstrap interval.",
            "",
            "**Fit time** is the median model time per origin times the "
            f"{len(origins)} origins of the full schedule, on one desktop CPU (machine B in "
            "[docs/methods.md](docs/methods.md)); the median, so that time lost to other "
            "work or to the machine sleeping is not counted. The statistical models were "
            "fitted three at a time and each is given a third. LightGBM is the only model "
            "given the holiday calendar. N-HiTS is refitted every fourth origin and "
            "forecasts every origin from its latest weights. PatchTST and TimesFM are not "
            "built yet.",
            "",
            f"**Reconciliation: {best} with MinT**",
            "",
            f"Largest coherence breach over every origin and horizon step, in incidents: "
            f"{base_breach:.1f} before reconciling, {mint.coherence_error:.1e} after.",
            "",
            tables.markdown_table(
                [
                    "Level",
                    f"CRPS, {best}",
                    f"CRPS, {best} + MinT",
                    "Change from reconciling",
                    "Skill of reconciling",
                ],
                reconcile_rows,
            ),
            "",
            "Only the medians are reconciled; each node's quantiles move with its median. "
            "Probabilistic reconciliation is not built yet.",
            "",
            "**The rota: staffing every dispatch area, priced against an oracle**",
            "",
            f"Inputs are illustrative and replaceable ([inputs/decision.toml]"
            f"(inputs/decision.toml)): one unit handles {inputs.demand_per_unit:g} incidents a "
            f"day, a spare unit-day costs {inputs.cost_over:g} and a missing one "
            f"{inputs.cost_under:g}, so the costs imply staffing at the {implied:.0%} "
            f"quantile. The oracle, knowing each day's demand, staffs "
            f"{staffing.oracle_units:.1f} units a day and costs nothing. Units and cost are "
            f"summed over the {len(hierarchy.rows_at('area'))} areas and averaged over the "
            f"{origins.horizon} days ahead.",
            "",
            tables.markdown_table(
                [
                    "Service level",
                    "Method",
                    "Units staffed per day",
                    "Realised cost per day against the oracle",
                    f"Cost minus {best}",
                ],
                staffing_rows,
            ),
        ]
    )
    typer.echo(block)
    if write:
        try:
            README.write_text(
                tables.splice(README.read_text(encoding="utf-8"), block),
                encoding="utf-8",
                newline="\n",
            )
        except ValueError as error:
            typer.echo(str(error))
            raise typer.Exit(code=1) from error
        typer.echo(f"wrote {README}")


def _names(option: str) -> list[str]:
    """Split a comma-separated option into names.

    Args:
        option: The option's value.

    Returns:
        The non-blank names, stripped, in order.
    """
    return [name.strip() for name in option.split(",") if name.strip()]


def _method_label(name: str) -> str:
    """Name a checkpoint the way the tables do.

    Args:
        name: A checkpoint name, such as ``N-HiTS-refit4``.

    Returns:
        For example ``N-HiTS, refitted every 4 weeks`` or ``LightGBM, global``.
    """
    if name == "LightGBM":
        return "LightGBM, global"
    model, _, refit = name.partition("-refit")
    if refit.isdigit():
        return f"{model}, refitted every {refit} weeks"
    return name


def _skill_rows(
    label: str,
    scores: Scores,
    naive: Scores,
    seconds: float,
    window_origins: int,
    listed: list[Origin],
    start: int,
) -> list[list[str]]:
    """One results row per hierarchy level for one method.

    Args:
        label: The method's name in the table.
        scores: Its scores, from ``start`` onward.
        naive: Seasonal naive on the same origins.
        seconds: Its compute time for the full schedule.
        window_origins: Origins in the worst-coverage window.
        listed: Every origin in the schedule.
        start: The first origin scored.

    Returns:
        Three rows of cells.
    """
    from headroom.report import tables
    from headroom.score.coverage import level_index

    nominal_at = list(scores.nominal).index(MAIN_NOMINAL)
    median_at = level_index(lv.REPORTING, 0.5)
    rows = []
    for i, (level, level_name) in enumerate(LEVEL_NAMES.items()):
        crps = scores.by_origin(scores.crps, level)
        pinball = scores.by_origin(scores.pinball[..., median_at], level)
        if scores is naive:
            crps_skill = pinball_skill = "reference"
        else:
            crps_skill = tables.number(
                tables.Estimate.of(skill_interval(crps, naive.by_origin(naive.crps, level))),
                places=3,
                signed=True,
            )
            naive_pinball = naive.by_origin(naive.pinball[..., median_at], level)
            pinball_skill = tables.number(
                tables.Estimate.of(skill_interval(pinball, naive_pinball)),
                places=3,
                signed=True,
            )
        hits = scores.by_origin(scores.hits[..., nominal_at].astype(np.float64), level)
        at, worst = tables.worst_window(hits, window_origins)
        widths = scores.by_origin(scores.widths[..., nominal_at], level)
        rows.append(
            [
                label if i == 0 else "",
                level_name,
                crps_skill,
                pinball_skill,
                tables.number(tables.Estimate.of(confidence_interval(hits)), places=3),
                f"{worst:.3f} ({listed[start + at].day})",
                tables.number(tables.Estimate.of(confidence_interval(widths))),
                tables.duration(seconds) if i == 0 else "",
            ]
        )
    return rows


def _from_origin(scores: Scores, start: int) -> Scores:
    """Restrict a backtest's scores to the origins from ``start`` onward.

    Args:
        scores: Scores over every origin.
        start: The first origin to keep.

    Returns:
        The same scores with every per-origin array sliced. Compute time is kept whole,
        because it is what the full run cost.
    """
    from dataclasses import replace

    if start == 0:
        return scores
    return replace(
        scores,
        crps=scores.crps[start:],
        pinball=scores.pinball[start:],
        absolute_error=scores.absolute_error[start:],
        hits=scores.hits[start:],
        widths=scores.widths[start:],
        reporting_quantiles=scores.reporting_quantiles[start:],
    )


def _print_scores(scores: Scores, baseline: Scores | None, against: Scores | None) -> None:
    """Print one model's scores per hierarchy level, each with its interval.

    Args:
        scores: The model's scores.
        baseline: Seasonal naive on the same origins, or None when printing the baseline.
        against: A model to difference CRPS against per origin, or None.
    """
    nominal_at = list(scores.nominal).index(MAIN_NOMINAL)
    typer.echo(f"{scores.model}  model time {scores.fit_seconds / 3600:.2f}h")
    for level in ("city", "borough", "area"):
        crps = scores.by_origin(scores.crps, level)
        point, low, high = confidence_interval(crps)
        line = f"  {level:8} CRPS {point:8.3f} [{low:7.3f}, {high:7.3f}]"
        if baseline is not None:
            s, s_low, s_high = skill_interval(crps, baseline.by_origin(baseline.crps, level))
            line += f"  skill {s:+.4f} [{s_low:+.4f}, {s_high:+.4f}]"
        if against is not None:
            d, d_low, d_high = confidence_interval(
                crps - against.by_origin(against.crps, level)
            )
            line += f"  minus {against.model} {d:+.3f} [{d_low:+.3f}, {d_high:+.3f}]"
        cov, cov_low, cov_high = confidence_interval(
            scores.by_origin(scores.hits[..., nominal_at].astype(np.float64), level)
        )
        wid, wid_low, wid_high = confidence_interval(
            scores.by_origin(scores.widths[..., nominal_at], level)
        )
        line += (
            f"  coverage90 {cov:.4f} [{cov_low:.4f}, {cov_high:.4f}]"
            f"  width {wid:8.2f} [{wid_low:8.2f}, {wid_high:8.2f}]"
        )
        typer.echo(line)


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
