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
    headroom neural              N-HiTS or PatchTST, refitted on a schedule (hours; resumable)
    headroom zeroshot            TimesFM 2.5, pretrained, nothing fitted (hours; resumable)
    headroom reconcile           MinT on a model's medians: coherence and the change in CRPS
    headroom decide              staffing from each forecast, priced against an oracle
    headroom score               score finished checkpoints as skill against the baseline
    headroom report              fill the README's results tables (the only thing that may)
    headroom charts              the coverage chart and the fan chart, as PNG
    headroom export              the dashboard's JSON, validated against its schema
    headroom serve               the dashboard locally, with the headers the host sends

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
from typing import TYPE_CHECKING, Annotated, Any, Final

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
from headroom.models import foundation
from headroom.models.baselines import SeasonalNaive
from headroom.reconcile.mint import ReconciledMedians
from headroom.score import levels as lv
from headroom.score.bootstrap import BLOCK, confidence_interval, skill_interval
from headroom.score.coverage import empirical_coverage
from headroom.types import LEVELS, Level

if TYPE_CHECKING:  # Imported for the annotation only; matplotlib is not loaded to type it.
    from headroom.report.charts import CoverageSeries

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

#: Models scored on a conformal distribution built from their own past errors rather than
#: on quantiles of their own, and the grid each one's checkpoint holds. LightGBM forecasts
#: a median and nothing else; TimesFM forecasts the deciles, which do not reach the 0.025
#: and 0.975 this project reports, so the median is taken from them and the rest is kept
#: only to report the one interval the model can produce unaided (`docs/methods.md`).
STORED_LEVELS: Final[dict[str, npt.NDArray[np.float64]]] = {
    "LightGBM": np.array([0.5]),
    foundation.NAME: foundation.DECILES,
}
POINT_MODELS = frozenset(STORED_LEVELS)

#: Models PLAN.md section 1 promises a row for, as (network name, label when not built).
#: A row is written as "not built" only while no checkpoint for that network is reported,
#: so a model cannot appear as both run and unbuilt.
PROMISED_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("N-HiTS", "N-HiTS"),
    ("PatchTST", "PatchTST"),
    ("TimesFM", "TimesFM, zero-shot"),
)


def _stored_levels(name: str) -> npt.NDArray[np.float64]:
    """The quantile grid a model's checkpoint holds.

    Args:
        name: The model's checkpoint name.

    Returns:
        The model's own grid if it has one in :data:`STORED_LEVELS`, else the scoring grid.
    """
    return STORED_LEVELS.get(name, lv.SCORING)


def _median_at(levels: npt.NDArray[np.float64]) -> int:
    """Where the median sits on a quantile grid.

    Args:
        levels: A quantile grid.

    Returns:
        The index of the level closest to 0.5.
    """
    return int(np.abs(levels - 0.5).argmin())


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
        _stored_levels(model.name),
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
    model_name: Annotated[
        str, typer.Option("--model", help="The network: N-HiTS or PatchTST.")
    ] = "N-HiTS",
) -> None:
    """Back-test a neural model, refitting on a schedule and forecasting every origin.

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
        model_name: The network, N-HiTS or PatchTST.

    Raises:
        typer.Exit: The network is not one this command builds.
    """
    import logging

    from headroom.backtest.store import open_checkpoint
    from headroom.models.neural import DEFAULTS, GlobalNeural

    if model_name not in DEFAULTS:
        typer.echo(f"unknown network {model_name!r}; choose from {sorted(DEFAULTS)}")
        raise typer.Exit(code=1)

    warnings.filterwarnings("ignore")
    for noisy in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    panel = _panel()
    origins = Origins(
        days=panel.days, step=step, train_window=window or None, refit_every=refit_every
    )
    model = GlobalNeural(levels=lv.SCORING, horizon=origins.horizon, name=model_name)
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
def zeroshot(
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    save_every: Annotated[int, typer.Option(help="Origins between checkpoint writes.")] = 25,
) -> None:
    """Run TimesFM 2.5 zero-shot over every origin, resumably.

    Nothing is trained: the weights are Google's, downloaded once, and each origin is
    inference over the same 1,095-day window every other model gets. The checkpoint holds
    the model's nine deciles; `headroom score` takes the median from them and builds the
    distribution it is scored on from its own past errors.

    **This run covers the whole backtest, and most of it is exposed to the leak** described
    in PLAN.md section 2.7a: the weights postdate those origins. The result is the clean
    window, `headroom score --from-day`; the exposed numbers are labelled and never pooled
    with it.

    Args:
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
        save_every: Origins between checkpoint writes.
    """
    from headroom.backtest.store import open_checkpoint
    from headroom.models.foundation import PRETRAINING_ENDS, ZeroShotTimesFM, clean_origins

    panel, origins = _schedule(step, window)
    model = ZeroShotTimesFM(horizon=origins.horizon, context=window or TRAIN_WINDOW_DAYS)
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = open_checkpoint(
        out / _checkpoint_name(model.name, step, origins),
        model.name,
        origins,
        panel.hierarchy.n_nodes,
        _stored_levels(model.name),
    )
    listed = list(origins)
    clean = clean_origins([origin.day for origin in listed], PRETRAINING_ENDS)
    typer.echo(
        f"{len(origins)} origins at step {step}, {model.name} zero-shot, output {out}\n"
        f"clean window: origins {clean} to {len(origins) - 1} ({listed[clean].day} onward), "
        f"{len(origins) - clean} of {len(origins)}; the rest are exposed to the leak"
    )
    typer.echo(f"resuming from origin {checkpoint.n_done} of {len(origins)}")

    started = time.perf_counter()
    model.load()
    typer.echo(f"loaded and compiled in {time.perf_counter() - started:.0f}s")

    started = time.perf_counter()
    computed = 0
    for origin in origins:
        if checkpoint.done[origin.number]:
            continue
        began = time.perf_counter()
        deciles = model.forecast(
            panel.values[:, origin.train], origins.horizon, _stored_levels(model.name)
        )
        elapsed = time.perf_counter() - began
        checkpoint.record(origin.number, deciles, elapsed)
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
def reconcile(
    model: Annotated[str, typer.Option(help="A model with a finished checkpoint.")] = "ETS",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
) -> None:
    """Reconcile a model with MinT, both ways, and report coherence and the change in CRPS.

    ``W`` comes from the model's own errors at the same horizon step over the previous 52
    origins, so the first 53 origins are not reconciled and every number here is on
    origins 53 onward. Two reconciliations are reported from one pass:

    **+ MinT** reconciles the median and moves each node's quantiles with it. The medians
    are coherent; the spread is the base model's and nothing is claimed about it.

    **+ MinT paths** is the probabilistic reconciliation of `PLAN.md` section 2.5. Each of
    the 52 error vectors in the window is added to the base forecast and put through the
    same projection, so every draw is coherent across all 37 series at once and the
    distribution is read off those draws. Its quantiles still do not sum, and
    `headroom.reconcile.paths` says why that is correct rather than a defect.

    A model that forecasts only a median is reconciled the same way, because the paths take
    their spread from the error window rather than from the model's own quantiles. Its base
    row is its conformal distribution, which is what `score` gives it.

    Args:
        model: The model whose finished checkpoint is reconciled.
        step: Days between origins the checkpoint was built with.
        window: Trailing training window it was built with, or 0 for expanding.

    Raises:
        typer.Exit: The checkpoint is missing or incomplete.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.reconcile.mint import shift_quantiles
    from headroom.reconcile.paths import reconcile_paths

    panel, origins = _schedule(step, window)
    checkpoint = _complete_checkpoint(model, step, origins, panel.hierarchy.n_nodes)
    actual = np.stack([panel.values[:, origin.target] for origin in origins])
    base_median = checkpoint.quantiles[..., _median_at(_stored_levels(model))]

    started = time.perf_counter()
    result = reconcile_paths(base_median, actual, panel.hierarchy, lv.SCORING, step)
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
        f"medians {result.coherence_error:.2e}, and the same over every sample path\n"
        f"  shrinkage intensity: median {np.median(intensity):.3f}, "
        f"range {intensity.min():.3f} to {intensity.max():.3f}\n"
        f"  path values below zero, not floored so that coherence stays exact: "
        f"{result.negative_share:.4%}"
    )

    seconds = float(checkpoint.seconds.sum())
    base_quantiles = _distribution(model, checkpoint.quantiles, actual, step)
    base = score_forecasts(
        model,
        panel.hierarchy,
        origins,
        base_quantiles[start:],
        actual[start:],
        lv.SCORING,
        seconds,
    )
    shifted = shift_quantiles(
        base_quantiles[start:], base_median[start:], result.medians[start:]
    )
    del checkpoint, base_quantiles
    reconciled = score_forecasts(
        f"{model} + MinT",
        panel.hierarchy,
        origins,
        shifted,
        actual[start:],
        lv.SCORING,
        seconds,
    )
    del shifted
    coherent = score_forecasts(
        f"{model} + MinT paths",
        panel.hierarchy,
        origins,
        result.quantiles[start:],
        actual[start:],
        lv.SCORING,
        seconds,
    )
    _print_scores(base, None, None)
    _print_scores(reconciled, None, base)
    _print_scores(coherent, None, base)


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
    checkpoint = open_checkpoint(path, name, origins, n_nodes, _stored_levels(name))
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

    A model in :data:`STORED_LEVELS` is given the conformal predictive distribution around
    its own median, from its own past errors, which is ``nan`` before the first origin with
    a full calibration window.

    Args:
        name: The model's name.
        quantiles: Its checkpoint's forecasts, on the grid :data:`STORED_LEVELS` gives it.
        actual: What happened, shape ``(n_origins, n_nodes, horizon)``.
        step: Days between origins.

    Returns:
        Quantiles, shape ``(n_origins, n_nodes, horizon, n_scoring)``.
    """
    from headroom.conformal.predictive import predictive_quantiles

    if name not in POINT_MODELS:
        return quantiles
    median = quantiles[..., _median_at(_stored_levels(name))]
    return predictive_quantiles(median, actual, lv.SCORING, step).quantiles


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
    from_day: Annotated[
        str, typer.Option(help="Score only origins on or after this day, ISO format.")
    ] = "",
    pinball: Annotated[
        bool,
        typer.Option(help="Also print pinball skill at every reporting quantile, as markdown."),
    ] = False,
) -> None:
    """Score finished checkpoints as skill against seasonal naive, and paired.

    Seasonal naive is rerun on the same origins, so every comparison is paired. Each
    number carries a 95 percent block-bootstrap interval.

    Coverage for the statistical models is of their own prediction quantiles, with no
    conformal step, so no coverage guarantee is claimed. A model whose checkpoint does not
    hold the scoring grid (LightGBM, TimesFM) is given a conformal predictive distribution
    around its own median from its own past errors (`headroom.conformal.predictive`),
    which exists only from the first origin with a full calibration window. When one is
    included, **every model is scored on the origins from that one onward**, so the tables
    stay paired and the numbers differ from a run without it.

    ``--from-day`` is how TimesFM's clean window is scored: the origins whose whole horizon
    falls after its pretraining data ends (PLAN.md section 2.7a). It moves every model
    listed onto that window, because a comparison across two different sets of origins is
    not a comparison.

    Args:
        models: Comma-separated names of models whose checkpoints are complete.
        against: The model each other model's CRPS is differenced against, per origin.
        step: Days between origins the checkpoints were built with.
        window: Trailing training window they were built with, or 0 for expanding.
        from_day: Score only origins on or after this day.
        pinball: Also print pinball skill at each reporting quantile, per level, as a
            markdown table. CRPS is twice the integral of the pinball loss over all
            quantiles, so this is the same measurement read across the distribution
            instead of summed: it says where in the distribution a model wins.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete, or no origin is that late.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.conformal.predictive import first_valid_origin

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
    if from_day:
        # The conformal window still ends at the origin before each forecast, so a later
        # start drops origins from the front of the scores and changes nothing about how
        # any forecast was produced.
        wanted_from = date.fromisoformat(from_day)
        later = [origin.number for origin in listed if origin.day >= wanted_from]
        if not later:
            typer.echo(
                f"no origin falls on or after {wanted_from}; the last is {listed[-1].day}"
            )
            raise typer.Exit(code=1)
        start = max(start, later[0])
    typer.echo(
        f"origins {start} to {len(origins) - 1} of {len(origins)} at step {step} "
        f"({listed[start].day} to {listed[-1].day}), output {out}"
    )
    if point_models:
        typer.echo(
            f"{_listing(point_models)}: scored on a conformal distribution around the "
            "median, built from its own errors over the previous 52 origins"
        )

    actual = np.stack([panel.values[:, origin.target] for origin in origins])
    baseline_scores = _from_origin(
        run(SeasonalNaive(), "Seasonal naive", panel.values, panel.hierarchy, origins), start
    )
    everything: dict[str, Scores] = {}
    for name in wanted:
        checkpoint = _complete_checkpoint(name, step, origins, panel.hierarchy.n_nodes)
        quantiles = _distribution(name, checkpoint.quantiles, actual, step)
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
    if pinball:
        for scores in everything.values():
            _print_pinball(scores, baseline_scores)


#: Where the charts are written, and the period they shade as the demand shift.
CHARTS = Path("docs/charts")
SHIFT = (date(2020, 3, 1), date(2020, 6, 1))


def _level(name: str) -> Level:
    """Narrow a level option to one of the hierarchy's own levels.

    Args:
        name: The option's value.

    Returns:
        The level.

    Raises:
        typer.Exit: It is not a level of this hierarchy.
    """
    for level in LEVELS:
        if level == name:
            return level
    typer.echo(f"{name} is not a level; choose one of {', '.join(LEVELS)}")
    raise typer.Exit(code=1)


def _coverage_series(
    panel: Panel, origins: Origins, alphas: list[float], level: Level
) -> list["CoverageSeries"]:
    """Wrap seasonal naive in every conformal method and measure coverage through time.

    Shared by the chart and the export so the PNG in the README and the panel on the
    dashboard cannot be drawn from different numbers. The backtest and its wrappers are
    hundreds of megabytes and are dropped here, before anything else is opened.

    Args:
        panel: The panel.
        origins: The schedule.
        alphas: Target miscoverages, so 0.10 is a 90 percent interval.
        level: Hierarchy level coverage is measured at.

    Returns:
        One series per method per nominal level, in that order.
    """
    from headroom.report.charts import CoverageSeries

    typer.echo("running seasonal naive for the coverage panel")
    naive = run(SeasonalNaive(), "Seasonal naive", panel.values, panel.hierarchy, origins)
    series: list[CoverageSeries] = []
    for target in alphas:
        # The legend carries the method, not its alpha: each row of the chart is one
        # nominal level and says so on its own axis, so repeating it truncates the labels
        # and tells the reader nothing.
        for method, label in (
            (SplitConformal(target), "split"),
            (AdaptiveConformal(target, gamma=0.05), "adaptive, gamma 0.05"),
            (AggregatedConformal(target), "aggregated, 6 experts"),
        ):
            result = apply(method, naive, panel.values, target)
            series.append(
                CoverageSeries(
                    method=label,
                    nominal=1.0 - target,
                    coverage=result.coverage_by_origin(level),
                    width=result.width_by_origin(level),
                )
            )
            typer.echo(f"  {1 - target:.0%} {method.name[:44]}: {result.coverage(level):.4f}")
            del result
    return series


@app.command()
def charts(
    model: Annotated[str, typer.Option(help="The model the fan chart draws.")] = "ETS",
    alpha: Annotated[
        str, typer.Option(help="Target miscoverages for the coverage chart.")
    ] = "0.20,0.10,0.05",
    level: Annotated[
        str, typer.Option(help="Hierarchy level for the coverage chart.")
    ] = "city",
    horizon_step: Annotated[int, typer.Option(help="Days ahead the fan chart draws.")] = 14,
    first_day: Annotated[str, typer.Option(help="First day of the fan chart.")] = "2019-09-01",
    last_day: Annotated[str, typer.Option(help="Last day of the fan chart.")] = "2020-09-01",
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    out: Annotated[str, typer.Option(help="Directory the PNGs are written to.")] = str(CHARTS),
) -> None:
    """Draw the coverage chart and the fan chart `PLAN.md` section 1 promises.

    The coverage chart wraps seasonal naive in each conformal method at each nominal level
    and plots trailing coverage against nominal, with the mean width of the same intervals
    underneath, because a method reaches nominal trivially by being wide enough. This is
    the picture of the project's headline finding: adaptive conformal cannot widen past
    the largest nonconformity score in its calibration window, so it does not recover
    inside the 2020 shift however fast it adapts.

    The fan chart draws one finished model's bands at a fixed horizon step against what
    happened, one panel per hierarchy level, over a window that contains the shift.

    Args:
        model: The model whose checkpoint the fan chart draws.
        alpha: Comma-separated target miscoverages, so 0.10 is a 90 percent interval.
        level: Hierarchy level the coverage chart measures.
        horizon_step: Days ahead the fan chart draws, between 1 and the horizon.
        first_day: First target day of the fan chart, ISO format.
        last_day: Last target day of the fan chart, ISO format.
        step: Days between origins.
        window: Trailing training window in days, or 0 to let it expand.
        out: Directory the PNGs are written to.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete, or the options select no days.
    """
    from headroom.report.charts import coverage_chart, fan_chart

    panel, origins = _schedule(step, window)
    listed = list(origins)
    directory = Path(out)
    window_origins = max(3, SHIFT_WINDOW_DAYS // step)

    series = _coverage_series(panel, origins, [float(a) for a in _names(alpha)], _level(level))
    written = coverage_chart(
        series,
        [origin.day for origin in listed],
        directory / "coverage.png",
        window=window_origins,
        level_name=level,
        shift=SHIFT,
    )
    typer.echo(f"wrote {written}")
    # The series are hundreds of megabytes and the fan chart needs none of them, so they go
    # before a checkpoint of the same size is opened.
    del series

    checkpoint = _complete_checkpoint(model, step, origins, panel.hierarchy.n_nodes)
    quantiles = _distribution(model, checkpoint.quantiles, actual_of(panel, origins), step)
    if not 1 <= horizon_step <= origins.horizon:
        typer.echo(f"horizon step {horizon_step} is outside 1 to {origins.horizon}")
        raise typer.Exit(code=1)
    wanted = (date.fromisoformat(first_day), date.fromisoformat(last_day))
    # One row per origin, read at a fixed horizon step, so the panels are a series through
    # time rather than a single origin's fourteen days.
    rows = [
        origin.number
        for origin in listed
        if wanted[0] <= panel.days[origin.index + horizon_step] <= wanted[1]
        and np.isfinite(quantiles[origin.number]).all()
    ]
    if not rows:
        typer.echo(f"no origin forecasts a day between {wanted[0]} and {wanted[1]}")
        raise typer.Exit(code=1)
    picked = [panel.hierarchy.rows_at(name)[0] for name in LEVEL_NAMES]
    labels = [
        f"{LEVEL_NAMES[name]}: {panel.hierarchy.nodes[row]}"
        for name, row in zip(LEVEL_NAMES, picked, strict=True)
    ]
    # Three series at one horizon step out of 37 series at fourteen: take the slice and let
    # the checkpoint go before drawing anything.
    bands = quantiles[np.ix_(rows, picked)][:, :, horizon_step - 1, :].copy()
    del checkpoint, quantiles
    written = fan_chart(
        bands,
        lv.SCORING,
        np.stack([panel.values[picked, listed[row].index + horizon_step] for row in rows]),
        [panel.days[listed[row].index + horizon_step] for row in rows],
        labels,
        directory / "fan.png",
        horizon_step=horizon_step,
        shift=SHIFT,
    )
    typer.echo(f"wrote {written}")


def actual_of(panel: Panel, origins: Origins) -> npt.NDArray[np.float64]:
    """Stack what happened over every origin's horizon.

    Args:
        panel: The panel.
        origins: The schedule.

    Returns:
        Shape ``(n_origins, n_nodes, horizon)``.
    """
    return np.stack([panel.values[:, origin.target] for origin in origins])


#: The file `headroom report` writes its tables into.
README = Path("README.md")

#: Hierarchy levels in table order, with the names the tables use. Typed as the hierarchy's
#: own level literals so that a key can be handed straight to ``rows_at``.
LEVEL_NAMES: Final[dict[Level, str]] = {
    "city": "City",
    "borough": "Borough",
    "area": "Dispatch area",
}

#: Days in the rolling window the worst-coverage column is measured over.
SHIFT_WINDOW_DAYS = 91

#: Origins a window needs before a block-bootstrap interval is reported on it. Each
#: resample is laid out as whole blocks, so a window of a few blocks resamples from almost
#: nothing: on TimesFM's 40-origin cross-check window the interval came back not even
#: containing its own point estimate (`docs/methods.md`). Four blocks is the floor, and
#: below it the tables print the difference with no interval and say why.
MIN_BOOTSTRAP_ORIGINS: Final[int] = 4 * BLOCK


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
    zero_shot: Annotated[
        str,
        typer.Option(
            "--zero-shot", help="Pretrained checkpoints, reported on their own windows."
        ),
    ] = foundation.NAME,
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

    A pretrained model named in ``--zero-shot`` is kept out of that table on purpose. Its
    weights postdate most of the backtest, so its numbers there would not mean what the
    other rows mean, and PLAN.md section 2.7a forbids pooling them. It gets its own table,
    one block of rows per window, with the exposed window labelled as exposed.

    Args:
        statistical: Statistical models to choose the best from, by CRPS skill averaged
            over the three levels. The best is also reconciled and staffed from.
        boosting: The gradient-boosting model's checkpoint name.
        neural: Neural checkpoint names, such as ``N-HiTS-refit4``.
        zero_shot: Pretrained checkpoints, scored on their own windows and never pooled
            with the main table. Skipped where the checkpoint does not exist.
        step: Days between origins the checkpoints were built with.
        window: Trailing training window they were built with, or 0 for expanding.
        write: Write the tables into README.md between its report markers.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete, or the README has no markers.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.conformal.predictive import first_valid_origin
    from headroom.decide.newsvendor import critical_ratio, load_inputs
    from headroom.reconcile.mint import shift_quantiles
    from headroom.reconcile.paths import reconcile_paths
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
    median_at = int(np.abs(lv.SCORING - 0.5).argmin())
    base_median = checkpoint.quantiles[..., median_at]
    mint = reconcile_paths(base_median, actual, hierarchy, lv.SCORING, step)
    if mint.first_valid > start:
        typer.echo(f"MinT starts at origin {mint.first_valid}, after {start}")
        raise typer.Exit(code=1)
    base_breach = max(
        hierarchy.coherence_error(base_median[o, :, h])
        for o in range(start, len(origins))
        for h in range(origins.horizon)
    )
    shifted = np.full_like(checkpoint.quantiles, np.nan)
    shifted[start:] = shift_quantiles(
        checkpoint.quantiles[start:], base_median[start:], mint.medians[start:]
    )
    del checkpoint
    reconciled = scored(f"{best} + MinT", shifted)
    demand[f"{best} + MinT"] = _area_demand(shifted, panel, start, service)
    del shifted
    # The probabilistic reconciliation: the decision layer was always meant to staff from a
    # reconciled distribution rather than from a reconciled median with borrowed spread.
    coherent_name = f"{best} + MinT paths"
    coherent = scored(coherent_name, mint.quantiles)
    demand[coherent_name] = _area_demand(mint.quantiles, panel, start, service)

    others: dict[str, Scores] = {}
    for name in [*_names(boosting), *_names(neural)]:
        checkpoint = _complete_checkpoint(name, step, origins, hierarchy.n_nodes)
        quantiles = _distribution(name, checkpoint.quantiles, actual, step)
        others[name] = scored(name, quantiles)
        seconds[name] = compute(checkpoint)
        demand[name] = _area_demand(quantiles, panel, start, service)
        del checkpoint, quantiles

    # A pretrained model is scored on its own windows and never pooled with the table
    # above, so it is built here rather than added to `others`.
    zero_shot_rows: list[list[str]] = []
    zero_shot_scored: set[str] = set()
    zero_shot_seconds: dict[str, float] = {}
    # Fitted models to difference against: the best statistical one and the global one. The
    # neural checkpoints are left out because three difference columns is a table nobody
    # reads, and the global model is the one that answers "did pretraining buy anything".
    zero_shot_against = [best, *(name for name in others if _refit_every(name) is None)]
    for name in _names(zero_shot):
        if not (output_dir() / _checkpoint_name(name, step, origins)).exists():
            continue
        checkpoint = _complete_checkpoint(name, step, origins, hierarchy.n_nodes)
        quantiles = _distribution(name, checkpoint.quantiles, actual, step)
        zero_shot_seconds[name] = compute(checkpoint)
        del checkpoint
        windows: list[tuple[str, int]] = []
        for label, first_day in foundation.WINDOWS:
            at = [origin.number for origin in listed if origin.day >= first_day]
            if at and max(at[0], start) < len(origins):
                windows.append((label, max(at[0], start)))
        windows.append(("Full backtest, exposed to the leak", start))
        for label, first_origin in windows:
            scores = score_forecasts(
                name,
                hierarchy,
                origins,
                quantiles[first_origin:],
                actual[first_origin:],
                lv.SCORING,
                0.0,
            )
            zero_shot_rows += _window_rows(
                label,
                listed[first_origin].day,
                scores,
                _from_origin(naive, first_origin - start),
                {
                    reference: _from_origin(
                        candidates[reference] if reference in candidates else others[reference],
                        first_origin - start,
                    )
                    for reference in zero_shot_against
                },
            )
            del scores
        zero_shot_scored.add(_base_model(name))
        del quantiles

    typer.echo("staffing")
    staffing = _staffing(demand, actual[start:], panel, service)
    window_origins = max(3, SHIFT_WINDOW_DAYS // step)

    methods: list[tuple[str, Scores, float]] = [
        ("Seasonal naive", naive, seconds["Seasonal naive"]),
        (f"Best statistical: {best}", candidates[best], seconds[best]),
    ]
    methods += [(_method_label(name), others[name], seconds[name]) for name in others]
    # Only promise a "not built" row for a network no checkpoint was reported for, so a
    # model that has since been run cannot appear twice with contradictory rows. A model
    # reported on its own windows is built but cannot share this table, because its
    # origins are not the origins every other row here is scored on.
    built = {_base_model(name) for name in others}
    absent = _absent_rows(built, zero_shot_scored)

    skill_rows: list[list[str]] = []
    for label, scores, cost in methods:
        skill_rows += _skill_rows(label, scores, naive, cost, window_origins, listed, start)
    skill_rows += [[label, note, "", "", "", "", "", ""] for label, note in absent]

    reconcile_rows = []
    for level, level_name in LEVEL_NAMES.items():
        before = candidates[best].by_origin(candidates[best].crps, level)
        places = min(3, tables.decimals_for(float(before.mean())) + 1)
        nominal_at = list(reconciled.nominal).index(MAIN_NOMINAL)
        row = [
            level_name,
            tables.number(tables.Estimate.of(confidence_interval(before))),
        ]
        for scores in (reconciled, coherent):
            after = scores.by_origin(scores.crps, level)
            hits = scores.by_origin(scores.hits[..., nominal_at].astype(np.float64), level)
            row += [
                tables.number(
                    tables.Estimate.of(confidence_interval(after - before)),
                    places=places,
                    signed=True,
                ),
                tables.number(tables.Estimate.of(confidence_interval(hits)), places=3),
            ]
        reconcile_rows.append(row)

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

    # Every note names the models actually scored, so none of them can go stale when a
    # model is added, renamed or run on a different schedule.
    # Base names, not table labels: a label carries its own comma ("N-HiTS, refitted every
    # 4 weeks") and two of those inside a list make the sentence unreadable.
    scored_own = [_base_model(name) for name in others if name not in POINT_MODELS]
    own_quantiles = f", {_listing(scored_own)}," if scored_own else ""
    conformal_note = _listing([_method_label(n) for n in others if n in POINT_MODELS])
    refits = [
        f"{_base_model(name)} every {every}"
        for name in others
        if (every := _refit_every(name)) is not None
    ]
    refit_note = (
        f" The neural models are refitted on a schedule, {_listing(refits)} origins, and "
        f"{'it forecasts' if len(refits) == 1 else 'each forecasts'} every origin from its "
        "latest weights."
        if refits
        else ""
    )
    missing = [
        network
        for network, _ in PROMISED_ROWS
        if network not in built and network not in zero_shot_scored
    ]
    verb = "is" if len(missing) == 1 else "are"
    unbuilt_note = f" {_listing(missing)} {verb} not built yet." if missing else ""

    zero_shot_block: list[str] = []
    if zero_shot_rows:
        named = _listing(sorted(zero_shot_scored))
        cost_note = _listing(
            [
                f"{name}, {tables.duration(zero_shot_seconds[name])}"
                for name in zero_shot_seconds
            ]
        )
        zero_shot_block = [
            f"**{named}, pretrained and used zero-shot: the windows it can be judged on**",
            "",
            f"{named} is not in the table above, and the reason is the result. Its weights "
            "were trained after most of these origins, on a corpus that contains the 2020 "
            "period in other series, so a forecast it makes of 2020 is not the same kind of "
            "claim as every other row's. It is scored here on three windows, **never "
            "pooled**: the clean window, whose whole horizon falls after the latest "
            "documented pretraining data; a shorter cross-check after the weights were "
            "published, which is a subset of the same forecasts; and the full backtest, "
            "which is labelled exposed and is not a result. It is trained on none of this "
            "data at all, and it is given the same window, horizon and scoring grid as "
            f"everything else. Inference over the full schedule: {cost_note}.",
            "",
            tables.markdown_table(
                [
                    "Window",
                    "Origins",
                    "Level",
                    "CRPS",
                    "CRPS skill",
                    *(f"CRPS minus {_method_label(name)}" for name in zero_shot_against),
                    "Coverage at 90% nominal",
                ],
                zero_shot_rows,
            ),
            "",
            f"Its forecast is its median and the distribution scored here is conformal from "
            "its own past errors, as LightGBM's is, because its own quantile head stops at "
            "the 0.1 and 0.9 quantiles and this table reports 95 percent intervals. A window "
            f"shorter than {MIN_BOOTSTRAP_ORIGINS} origins carries no bootstrap interval: "
            f"the block length is {BLOCK} origins, and a window of a few blocks resamples "
            "from too little to say anything. Those rows print the difference alone, and "
            "direction is all they carry.",
            "",
        ]

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
            f"of its own past errors), the statistical models{own_quantiles} are scored on "
            "their own quantiles with no conformal step, so no coverage is guaranteed for "
            f"them. {conformal_note} forecasts a median only, and its distribution "
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
            f"given the holiday calendar.{refit_note}{unbuilt_note}",
            "",
            *zero_shot_block,
            f"**Reconciliation: {best} with MinT, two ways**",
            "",
            f"Base forecasts made one series at a time do not sum: the largest breach of the "
            f"summing constraints is {base_breach:.1f} incidents. Both reconciliations close "
            f"it to {mint.coherence_error:.1e}, and **MinT paths closes it over every draw of "
            "the distribution**, not only the median.",
            "",
            tables.markdown_table(
                [
                    "Level",
                    f"CRPS, {best}",
                    "MinT: change in CRPS",
                    "MinT: coverage at 90%",
                    "MinT paths: change in CRPS",
                    "MinT paths: coverage at 90%",
                ],
                reconcile_rows,
            ),
            "",
            "**MinT** reconciles the median and moves each node's quantiles with it, so the "
            "medians are coherent and the spread is still the base model's. **MinT paths** "
            "is the probabilistic reconciliation: each of the 52 error vectors in the "
            "window is added to the base forecast and put through the same projection, so "
            "every draw is coherent across all 37 series at once and the distribution is "
            "read off those draws. Its quantiles still do not sum, and they are not supposed "
            "to: the boroughs do not have their bad days together, so the city's 90th "
            "percentile is below the sum of theirs. What is coherent is every draw, which is "
            "what a decision taken over the whole hierarchy needs. Paths are not floored at "
            f"zero, because that would break the coherence they exist for; "
            f"{mint.negative_share:.4%} of path values fall below zero.",
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


#: The static site, and the directory inside it the page fetches its JSON from.
DASHBOARD = Path("dashboard")
DASHBOARD_DATA = DASHBOARD / "data"

#: Service levels the slider stops at, beside whatever the decision inputs name. A staffing
#: table is only as continuous as the levels it was computed at, and each one costs another
#: pass over every origin, area and day, so the slider moves in whole percentage steps of
#: five rather than pretending to be continuous.
SLIDER_LEVELS: Final[tuple[float, ...]] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


@app.command()
def export(
    statistical: Annotated[
        str, typer.Option(help="The statistical model the page draws and reconciles.")
    ] = "ETS",
    boosting: Annotated[
        str, typer.Option(help="The gradient-boosting checkpoint.")
    ] = "LightGBM",
    neural: Annotated[
        str, typer.Option(help="Comma-separated neural checkpoints, staffed from.")
    ] = "N-HiTS-refit4,PatchTST-refit13",
    zero_shot: Annotated[
        str,
        typer.Option(
            "--zero-shot", help="Pretrained checkpoints, drawn on their own window only."
        ),
    ] = foundation.NAME,
    alpha: Annotated[
        str, typer.Option(help="Target miscoverages for the coverage panel.")
    ] = "0.20,0.10,0.05",
    level: Annotated[str, typer.Option(help="Hierarchy level the coverage panel measures.")] = (
        "city"
    ),
    horizon_step: Annotated[int, typer.Option(help="Days ahead the fan chart draws.")] = 14,
    step: Annotated[int, typer.Option(help="Days between origins.")] = 7,
    window: Annotated[
        int, typer.Option(help="Training window; 0 expands.")
    ] = TRAIN_WINDOW_DAYS,
    out: Annotated[str, typer.Option(help="Directory the JSON is written to.")] = str(
        DASHBOARD_DATA
    ),
) -> None:
    """Write the dashboard's JSON from the finished checkpoints.

    The page is static and has no backend to fail loudly, so everything it draws is computed
    here and checked against a schema before it is written. It is the same work `headroom
    report` does, on the same origins, from the same checkpoints: the dashboard is a second
    view of the README's numbers and not a second measurement of them.

    Takes several minutes and a few gigabytes of memory, one checkpoint at a time.

    Args:
        statistical: The statistical model the fan chart draws and the reconciliation view
            reconciles. One model, not a list: the page shows one forecast at a time.
        boosting: The gradient-boosting checkpoint, staffed from.
        neural: Neural checkpoints, staffed from.
        zero_shot: Pretrained checkpoints. They are drawn on their own window and never
            pooled with the other models, for the reason `PLAN.md` section 2.7a gives, so
            they are not staffed from either. Skipped where the checkpoint is missing.
        alpha: Comma-separated target miscoverages, so 0.10 is a 90 percent interval.
        level: Hierarchy level the coverage panel measures.
        horizon_step: Days ahead the fan chart draws, between 1 and the horizon.
        step: Days between origins the checkpoints were built with.
        window: Trailing training window they were built with, or 0 for expanding.
        out: Directory the JSON is written to.

    Raises:
        typer.Exit: A checkpoint is missing or incomplete, or an option is out of range.
    """
    from headroom.backtest.run import score_forecasts
    from headroom.conformal.predictive import first_valid_origin
    from headroom.decide.newsvendor import critical_ratio, load_inputs
    from headroom.reconcile.mint import shift_quantiles
    from headroom.reconcile.paths import reconcile_paths
    from headroom.report import export as ex

    panel, origins = _schedule(step, window)
    hierarchy = panel.hierarchy
    if not 1 <= horizon_step <= origins.horizon:
        typer.echo(f"horizon step {horizon_step} is outside 1 to {origins.horizon}")
        raise typer.Exit(code=1)
    start = first_valid_origin(origins.horizon, step)
    listed = list(origins)
    days = [origin.day for origin in listed[start:]]
    actual = actual_of(panel, origins)
    directory = Path(out)

    inputs = load_inputs()
    implied = critical_ratio(inputs.cost_over, inputs.cost_under)
    service = _service_levels(implied, (*inputs.service_levels, *SLIDER_LEVELS))

    # The coverage panel first, and then nothing of it is kept: the wrappers are hundreds of
    # megabytes and every checkpoint below is the same size again.
    covered = _coverage_series(panel, origins, [float(a) for a in _names(alpha)], _level(level))
    coverage = ex.coverage_section(
        level=level,
        days=days,
        window_origins=max(3, SHIFT_WINDOW_DAYS // step),
        shift=SHIFT,
        methods=[
            (one.method, one.nominal, one.coverage[start:], one.width[start:])
            for one in covered
        ],
    )
    del covered

    def scored(name: str, quantiles: npt.NDArray[np.float64]) -> Scores:
        typer.echo(f"scoring {name}")
        return score_forecasts(
            name, hierarchy, origins, quantiles[start:], actual[start:], lv.SCORING, 0.0
        )

    # The baseline the models panel takes skill against. It is cheap and it is not read
    # from a checkpoint, so it is run here rather than stored.
    typer.echo("running seasonal naive")
    naive_full = run(SeasonalNaive(), "Seasonal naive", panel.values, hierarchy, origins)
    naive = _from_origin(naive_full, start)
    nominal_at = list(naive.nominal).index(MAIN_NOMINAL)
    model_rows = [
        ex.model_row(
            name="Seasonal naive",
            kind="baseline",
            fit_seconds=naive_full.fit_seconds,
            levels=_model_levels(naive, None, nominal_at),
        )
    ]
    del naive_full

    typer.echo(f"reading {statistical}")
    checkpoint = _complete_checkpoint(statistical, step, origins, hierarchy.n_nodes)
    statistical_seconds = float(np.median(checkpoint.seconds)) * len(origins)
    band_at = [int(np.abs(lv.SCORING - q).argmin()) for q in ex.FAN_LEVELS]
    # (node, level, origin): the page draws one node at a time, so the node is the outer
    # axis and a series is one slice rather than a walk over the whole array.
    at_step = checkpoint.quantiles[start:, :, horizon_step - 1, :]
    bands = np.transpose(at_step[..., band_at], (1, 2, 0)).copy()
    del at_step
    forecast = ex.forecast_payload(
        model=statistical,
        horizon_step=horizon_step,
        days=[panel.days[origin.index + horizon_step] for origin in listed[start:]],
        nodes=list(hierarchy.nodes),
        node_levels=list(hierarchy.levels),
        actual=np.stack(
            [panel.values[:, origin.index + horizon_step] for origin in listed[start:]], axis=1
        ),
        bands=bands,
    )
    del bands

    base = scored(statistical, checkpoint.quantiles)
    model_rows.append(
        ex.model_row(
            name=statistical,
            kind="statistical",
            fit_seconds=statistical_seconds,
            levels=_model_levels(base, naive, nominal_at),
        )
    )
    demand = {
        "Seasonal naive": _naive_demand(panel, origins, start, service),
        statistical: _area_demand(checkpoint.quantiles, panel, start, service),
    }
    median_at = int(np.abs(lv.SCORING - 0.5).argmin())
    base_median = checkpoint.quantiles[..., median_at].copy()
    base_breach = max(
        hierarchy.coherence_error(base_median[o, :, h])
        for o in range(start, len(origins))
        for h in range(origins.horizon)
    )
    typer.echo(f"reconciling {statistical}")
    mint = reconcile_paths(base_median, actual, hierarchy, lv.SCORING, step)
    if mint.first_valid > start:
        typer.echo(f"MinT starts at origin {mint.first_valid}, after {start}")
        raise typer.Exit(code=1)
    shifted = np.full_like(checkpoint.quantiles, np.nan)
    shifted[start:] = shift_quantiles(
        checkpoint.quantiles[start:], base_median[start:], mint.medians[start:]
    )
    del checkpoint, base_median
    mint_name, paths_name = f"{statistical} + MinT", f"{statistical} + MinT paths"
    variants = [(mint_name, scored(mint_name, shifted))]
    demand[mint_name] = _area_demand(shifted, panel, start, service)
    del shifted
    variants.append((paths_name, scored(paths_name, mint.quantiles)))
    demand[paths_name] = _area_demand(mint.quantiles, panel, start, service)

    rows: list[dict[str, Any]] = []
    for name, label in LEVEL_NAMES.items():
        before = base.by_origin(base.crps, name)
        rows.append(
            ex.reconciliation_row(
                level=label,
                base_crps=confidence_interval(before),
                variants=[
                    (
                        variant,
                        confidence_interval(scores.by_origin(scores.crps, name) - before),
                        confidence_interval(
                            scores.by_origin(
                                scores.hits[..., nominal_at].astype(np.float64), name
                            )
                        ),
                    )
                    for variant, scores in variants
                ],
            )
        )
    reconciliation = ex.reconciliation_section(
        model=statistical,
        base_coherence_error=base_breach,
        reconciled_coherence_error=mint.coherence_error,
        negative_share=mint.negative_share,
        nominal=MAIN_NOMINAL,
        rows=rows,
    )
    del mint, variants, base

    for other_name in [*_names(boosting), *_names(neural)]:
        typer.echo(f"reading {other_name}")
        other = _complete_checkpoint(other_name, step, origins, hierarchy.n_nodes)
        other_seconds = float(np.median(other.seconds)) * len(origins)
        quantiles = _distribution(other_name, other.quantiles, actual, step)
        demand[_method_label(other_name)] = _area_demand(quantiles, panel, start, service)
        del other
        other_scores = scored(other_name, quantiles)
        model_rows.append(
            ex.model_row(
                name=_method_label(other_name),
                kind=_model_kind(other_name),
                fit_seconds=other_seconds,
                levels=_model_levels(other_scores, naive, nominal_at),
            )
        )
        del quantiles, other_scores

    # A pretrained model is scored on its own window and never pooled with the rows above:
    # its weights postdate most of these origins (`PLAN.md` section 2.7a). It is not
    # staffed from here either, for the same reason.
    own_windows: list[dict[str, Any]] = []
    for shot_name in _names(zero_shot):
        if not (output_dir() / _checkpoint_name(shot_name, step, origins)).exists():
            typer.echo(f"no {shot_name} checkpoint; leaving it off the models panel")
            continue
        typer.echo(f"reading {shot_name}")
        other = _complete_checkpoint(shot_name, step, origins, hierarchy.n_nodes)
        shot_seconds = float(np.median(other.seconds)) * len(origins)
        quantiles = _distribution(shot_name, other.quantiles, actual, step)
        del other
        window_label, first_day = foundation.WINDOWS[0]
        numbered = [origin.number for origin in listed if origin.day >= first_day]
        if not numbered or max(numbered[0], start) >= len(origins):
            typer.echo(f"{shot_name}: no clean window in this schedule")
            del quantiles
            continue
        first_origin = max(numbered[0], start)
        typer.echo(f"scoring {shot_name} on {len(origins) - first_origin} clean origins")
        shot_scores = score_forecasts(
            shot_name,
            hierarchy,
            origins,
            quantiles[first_origin:],
            actual[first_origin:],
            lv.SCORING,
            0.0,
        )
        own_windows.append(
            ex.own_window_row(
                name=dict(PROMISED_ROWS).get(_base_model(shot_name), shot_name),
                kind=_model_kind(shot_name),
                fit_seconds=shot_seconds,
                window=window_label,
                first=listed[first_origin].day,
                origins=len(origins) - first_origin,
                levels=_model_levels(
                    shot_scores, _from_origin(naive, first_origin - start), nominal_at
                ),
            )
        )
        del quantiles, shot_scores
    del naive

    typer.echo("staffing")
    staffing = _staffing(demand, actual[start:], panel, service)
    del demand
    reference = staffing.costs[statistical]
    methods = [
        ex.staffing_method(
            name=name,
            units=[confidence_interval(values) for values in staffing.units[name]],
            cost=[confidence_interval(values) for values in staffing.costs[name]],
            cost_change=[
                None if name == statistical else confidence_interval(values - reference[j])
                for j, values in enumerate(staffing.costs[name])
            ],
        )
        for name in staffing.costs
    ]

    payload = ex.dashboard_payload(
        generated=date.today(),
        origins=ex.origins_section(
            days=days,
            step=step,
            horizon=origins.horizon,
            train_window=origins.train_window,
        ),
        hierarchy=ex.hierarchy_section(
            nodes=list(hierarchy.nodes), levels=list(hierarchy.levels)
        ),
        coverage=coverage,
        models=ex.models_section(
            nominal=MAIN_NOMINAL,
            baseline="Seasonal naive",
            origins=len(days),
            rows=model_rows,
            own_windows=own_windows,
        ),
        reconciliation=reconciliation,
        staffing=ex.staffing_section(
            demand_per_unit=inputs.demand_per_unit,
            cost_over=inputs.cost_over,
            cost_under=inputs.cost_under,
            hours_per_unit_day=inputs.hours_per_unit_day,
            implied=implied,
            oracle_units=staffing.oracle_units,
            reference=statistical,
            service_levels=service,
            methods=methods,
        ),
    )
    try:
        for filename, one in ((ex.DASHBOARD, payload), (ex.FORECAST, forecast)):
            written = ex.write(directory, filename, one)
            typer.echo(f"wrote {written} ({written.stat().st_size / 1e6:.2f} MB)")
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8080,
    directory: Annotated[str, typer.Option(help="The site to serve.")] = str(DASHBOARD),
) -> None:
    """Serve the dashboard locally with the headers the host will send.

    `python -m http.server` sends none of the headers in `staticwebapp.config.json`, so it
    shows a page the content security policy would partly refuse. On a sister project that hid a
    broken chart on the live site for two weeks while every local check looked correct, so
    this serves the site the way it will be served.

    Args:
        port: Port to listen on.
        directory: The site to serve.

    Raises:
        typer.Exit: The directory has no `staticwebapp.config.json` to read headers from.
    """
    import json
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    site = Path(directory)
    config = site / "staticwebapp.config.json"
    if not config.exists():
        typer.echo(f"no {config}; run headroom export first, or name another directory")
        raise typer.Exit(code=1)
    declared = json.loads(config.read_text(encoding="utf-8"))
    headers: dict[str, str] = declared.get("globalHeaders", {})
    types: dict[str, str] = declared.get("mimeTypes", {})

    class Handler(SimpleHTTPRequestHandler):
        """A file server that sends the host's headers and the host's content types."""

        def guess_type(self, path: str | os.PathLike[str]) -> str:
            """Return the content type the host would send for a file.

            Python's own table does not know woff2, and a font served as a byte stream is
            one more way for the local page to differ from the published one.

            Args:
                path: The file being served.

            Returns:
                The type ``staticwebapp.config.json`` declares for that extension, or
                Python's own guess.
            """
            return types.get(Path(path).suffix) or super().guess_type(path)

        def end_headers(self) -> None:
            """Add the configured headers to every response."""
            for key, value in headers.items():
                self.send_header(key, value)
            super().end_headers()

    typer.echo(
        f"serving {site} on http://localhost:{port} with {len(headers)} headers "
        f"and {len(types)} content types from {config.name}"
    )
    address = ("127.0.0.1", port)
    with ThreadingHTTPServer(address, partial(Handler, directory=str(site))) as http:
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            typer.echo("stopped")


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


def _model_kind(name: str) -> str:
    """What sort of model a checkpoint holds, for the dashboard to group rows by.

    The page must not group by matching on names, so the kind travels in the payload. A
    checkpoint tagged with a refit schedule is one of the networks trained here; the
    pretrained model is named by :mod:`headroom.models.foundation`; the global boosted
    model forecasts a point and is given a conformal distribution.

    Args:
        name: A checkpoint name, such as ``PatchTST-refit13``.

    Returns:
        ``zero-shot``, ``neural``, ``boosting`` or ``statistical``.
    """
    base = _base_model(name)
    if base == foundation.NAME:
        return "zero-shot"
    if _refit_every(name) is not None:
        return "neural"
    if base == "LightGBM":
        return "boosting"
    return "statistical"


def _model_levels(
    scores: Scores, naive: Scores | None, nominal_at: int
) -> list[dict[str, Any]]:
    """One model's CRPS, skill, coverage and width at each hierarchy level.

    Args:
        scores: The model's scores, on the origins it is being reported on.
        naive: Seasonal naive on exactly those origins, or ``None`` for the baseline
            itself, whose skill against itself is zero by construction.
        nominal_at: Index of the reported nominal coverage in ``scores.nominal``.

    Returns:
        One entry per level, as :func:`headroom.report.export.model_level` shapes it.
    """
    from headroom.report import export as ex

    out = []
    for level, label in LEVEL_NAMES.items():
        crps = scores.by_origin(scores.crps, level)
        hits = scores.by_origin(scores.hits[..., nominal_at].astype(np.float64), level)
        widths = scores.by_origin(scores.widths[..., nominal_at], level)
        out.append(
            ex.model_level(
                level=label,
                crps=confidence_interval(crps),
                skill=(
                    None
                    if naive is None
                    else skill_interval(crps, naive.by_origin(naive.crps, level))
                ),
                coverage=confidence_interval(hits),
                width=confidence_interval(widths),
            )
        )
    return out


def _base_model(name: str) -> str:
    """The network's own name, without the refit schedule its checkpoint is tagged with.

    Args:
        name: A checkpoint name, such as ``PatchTST-refit13``.

    Returns:
        For example ``PatchTST``. A name with no schedule is returned unchanged.
    """
    return name.partition("-refit")[0]


def _refit_every(name: str) -> int | None:
    """The refit spacing a checkpoint name records, if it records one.

    Args:
        name: A checkpoint name, such as ``PatchTST-refit13``.

    Returns:
        The number of origins between refits, or None if the name carries no schedule.
    """
    refit = name.partition("-refit")[2]
    return int(refit) if refit.isdigit() else None


def _listing(items: list[str]) -> str:
    """Join names the way the notes read them out.

    Args:
        items: Names, already in the order they should be read.

    Returns:
        ``a``, ``a and b``, or ``a, b and c``. Empty for no items.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


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


def _absent_rows(built: set[str], own_windows: set[str]) -> list[tuple[str, str]]:
    """The promised rows with no scores in the main table, and what each says instead.

    PLAN.md section 1 promises a row per model. A model that has been run must never also
    be listed as not built, and a model reported on its own windows is run: it is out of
    the main table because its origins are not that table's origins, which is a different
    statement from not existing and has to read as one.

    Args:
        built: Network names with a checkpoint in the main table.
        own_windows: Network names reported on their own windows instead.

    Returns:
        ``(label, what the row says)`` for each promised network with no scores above.
    """
    return [
        (label, "own windows, below" if network in own_windows else "not built")
        for network, label in PROMISED_ROWS
        if network not in built
    ]


def _difference_cell(values: npt.NDArray[np.float64], places: int, signed: bool = True) -> str:
    """A paired difference with its interval, or the difference alone on a short window.

    Args:
        values: The paired per-origin differences.
        places: Decimal places.
        signed: Print a leading sign.

    Returns:
        The formatted cell. Below :data:`MIN_BOOTSTRAP_ORIGINS` origins the interval is
        left out rather than printed misleadingly narrow, and the note under the table
        says so.
    """
    from headroom.report import tables

    point = float(values.mean())
    if values.size < MIN_BOOTSTRAP_ORIGINS:
        return f"{point:+.{places}f}" if signed else f"{point:.{places}f}"
    return tables.number(
        tables.Estimate.of(confidence_interval(values)), places=places, signed=signed
    )


def _window_rows(
    window: str,
    first_day: date,
    scores: Scores,
    naive: Scores,
    references: dict[str, Scores],
) -> list[list[str]]:
    """One row per hierarchy level for one model on one window of origins.

    Args:
        window: How the table names the window.
        first_day: The window's first origin.
        scores: The model's scores on the window.
        naive: Seasonal naive on the same origins.
        references: Models to difference against, in column order, on the same origins.

    Returns:
        Three rows of cells, matching the header :func:`report` builds from the same
        reference names.
    """
    from headroom.report import tables

    nominal_at = list(scores.nominal).index(MAIN_NOMINAL)
    rows = []
    for i, (level, level_name) in enumerate(LEVEL_NAMES.items()):
        crps = scores.by_origin(scores.crps, level)
        hits = scores.by_origin(scores.hits[..., nominal_at].astype(np.float64), level)
        cells = [
            window if i == 0 else "",
            f"{crps.size} from {first_day}" if i == 0 else "",
            level_name,
            _difference_cell(
                crps, places=tables.decimals_for(float(crps.mean())), signed=False
            ),
        ]
        if crps.size < MIN_BOOTSTRAP_ORIGINS:
            cells.append(
                f"{1.0 - crps.mean() / naive.by_origin(naive.crps, level).mean():+.3f}"
            )
        else:
            cells.append(
                tables.number(
                    tables.Estimate.of(
                        skill_interval(crps, naive.by_origin(naive.crps, level))
                    ),
                    places=3,
                    signed=True,
                )
            )
        for reference in references.values():
            other = reference.by_origin(reference.crps, level)
            cells.append(
                _difference_cell(
                    crps - other,
                    places=min(3, tables.decimals_for(float(crps.mean())) + 1),
                )
            )
        cells.append(_difference_cell(hits, places=3, signed=False))
        rows.append(cells)
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


def _print_pinball(scores: Scores, baseline: Scores) -> None:
    """Print pinball skill at every reporting quantile, per level, as a markdown table.

    CRPS is twice the integral of the pinball loss over all quantiles, so the headline
    column and this table are the same measurement: one summed across the distribution and
    one read along it. The table is what says whether a model's win is in the middle, where
    a planner reads the median, or in the upper tail, where the rota is actually set.

    Args:
        scores: The model's scores.
        baseline: Seasonal naive on the same origins.
    """
    from headroom.report import tables

    rows = []
    for at, level in enumerate(lv.REPORTING):
        row = [f"{level:.3f}"]
        for name in LEVEL_NAMES:
            skill = skill_interval(
                scores.by_origin(scores.pinball[..., at], name),
                baseline.by_origin(baseline.pinball[..., at], name),
            )
            row.append(tables.number(tables.Estimate.of(skill), places=4, signed=True))
        rows.append(row)
    typer.echo(f"\n**{scores.model}: pinball skill against seasonal naive, per quantile**\n")
    typer.echo(tables.markdown_table(["Quantile", *LEVEL_NAMES.values()], rows))


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
