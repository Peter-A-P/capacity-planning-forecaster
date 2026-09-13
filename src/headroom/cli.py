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
"""

import time
import warnings
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from headroom.backtest.origins import TRAIN_WINDOW_DAYS, Origins
from headroom.backtest.run import run
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
from headroom.score.bootstrap import confidence_interval
from headroom.score.coverage import empirical_coverage

app = typer.Typer(add_completion=False, help=__doc__)
data_app = typer.Typer(help="Fetch and check the demand data.")
app.add_typer(data_app, name="data")

DATA = Path("data/nyc")
OUT = Path("backtest/out")
PANEL = DATA / "daily.parquet"

#: Nominal coverage the single-number tables report, matching PLAN.md section 1.
MAIN_NOMINAL = 0.90


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
    Stopping it is safe: every few origins are written to `backtest/out` and a rerun with
    the same options picks up where it left off.

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
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"step{step}-win{origins.train_window or 0}"
    points = {
        name: open_checkpoint(
            OUT / f"{name}-{suffix}.npz", name, origins, panel.hierarchy.n_nodes, lv.SCORING
        )
        for name in wanted
    }

    typer.echo(f"{len(origins)} origins at step {step}, models {batch.names}")
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


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
