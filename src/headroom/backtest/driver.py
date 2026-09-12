"""Run a backtest to a checkpoint, resuming whatever is already there.

:func:`headroom.backtest.run.run` holds a whole backtest in one call, which is right for
a model that takes milliseconds per origin. The statistical models take minutes, so this
drives the same loop with a checkpoint underneath it and can be stopped at any origin.

The forecasts it produces are identical either way: a model is handed the same training
slice whether it is the first origin of a fresh run or the resumed middle of an
interrupted one, because nothing is carried between origins.
"""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from headroom.backtest.origins import Origins
from headroom.backtest.run import Forecaster, Scores, score_forecasts
from headroom.backtest.store import Checkpoint, open_checkpoint
from headroom.hierarchy.spec import Hierarchy
from headroom.score import levels as levels_module

#: Origins between writes. Writing every origin doubles the wall clock on a cheap model
#: and buys nothing; ten origins is at most a few minutes of lost work on an expensive one.
SAVE_EVERY: int = 10


def run_to_checkpoint(
    model: Forecaster,
    name: str,
    panel_values: npt.NDArray[np.float64],
    hierarchy: Hierarchy,
    origins: Origins,
    checkpoint_path: Path,
    levels: npt.NDArray[np.float64] | None = None,
    save_every: int = SAVE_EVERY,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> Checkpoint:
    """Forecast every origin, resuming and checkpointing as it goes.

    Args:
        model: The forecaster.
        name: Its name, recorded in the checkpoint and compared on resume.
        panel_values: Node values, shape ``(n_nodes, n_days)``.
        hierarchy: The hierarchy the nodes belong to.
        origins: The origin schedule.
        checkpoint_path: Where to checkpoint.
        levels: The quantile grid. Defaults to the scoring grid, which contains the
            reporting grid.
        save_every: Origins between writes.
        on_progress: Called with ``(done, total, seconds_for_this_origin)`` after each
            origin, for a progress line.

    Returns:
        The completed checkpoint.

    Raises:
        ValueError: The panel does not match the hierarchy or the schedule.
    """
    grid = levels_module.SCORING if levels is None else levels
    if panel_values.shape[0] != hierarchy.n_nodes:
        raise ValueError(
            f"panel has {panel_values.shape[0]} nodes, hierarchy has {hierarchy.n_nodes}"
        )
    if panel_values.shape[1] != len(origins.days):
        raise ValueError(
            f"panel has {panel_values.shape[1]} days, the schedule has {len(origins.days)}"
        )

    checkpoint = open_checkpoint(checkpoint_path, name, origins, hierarchy.n_nodes, grid)
    since_save = 0

    for origin in origins:
        if checkpoint.done[origin.number]:
            continue
        started = time.perf_counter()
        forecast = model.forecast(panel_values[:, origin.train], origins.horizon, grid)
        elapsed = time.perf_counter() - started

        checkpoint.record(origin.number, forecast, elapsed)
        since_save += 1
        if since_save >= save_every:
            checkpoint.save()
            since_save = 0
        if on_progress is not None:
            on_progress(checkpoint.n_done, len(origins), elapsed)

    checkpoint.save()
    return checkpoint


def score_checkpoint(
    checkpoint: Checkpoint,
    name: str,
    panel_values: npt.NDArray[np.float64],
    hierarchy: Hierarchy,
    origins: Origins,
    levels: npt.NDArray[np.float64] | None = None,
) -> Scores:
    """Score a completed checkpoint's forecasts.

    Args:
        checkpoint: A complete checkpoint.
        name: The model's name.
        panel_values: The panel the backtest ran on.
        hierarchy: The hierarchy.
        origins: The origin schedule.
        levels: The grid the forecasts are on. Defaults to the scoring grid.

    Returns:
        The scores.

    Raises:
        ValueError: The checkpoint is not complete, so the scores would be partly ``nan``.
    """
    if not checkpoint.complete:
        raise ValueError(
            f"{checkpoint.n_done} of {checkpoint.done.size} origins are done; scoring a "
            "partial run would report a table over a period that was never forecast"
        )
    grid = levels_module.SCORING if levels is None else levels
    actual = np.stack([panel_values[:, origin.target] for origin in origins])
    return score_forecasts(
        name=name,
        hierarchy=hierarchy,
        origins=origins,
        quantiles=checkpoint.quantiles,
        actual=actual,
        scoring=grid,
        fit_seconds=float(checkpoint.seconds.sum()),
    )
