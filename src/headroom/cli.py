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
    headroom score               score finished checkpoints as skill against the baseline

Backtest output goes to `backtest/out` unless `HEADROOM_OUT` names another directory. A
full weekly checkpoint is about 740 MB per model, which is worth keeping out of a synced
folder.
"""

import os
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from headroom.backtest.origins import TRAIN_WINDOW_DAYS, Origins
from headroom.backtest.run import Scores, run
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
