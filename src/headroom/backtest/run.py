"""Run a forecaster over every origin and score it.

## Why this scores as it goes

The obvious design keeps every quantile forecast and scores them at the end. On this
panel that is 963 origins by 37 nodes by a 14-day horizon by 199 scoring levels, which is
794 MB for one model, before any of the four methods the tables compare. So the scoring
grid is consumed at the origin that produced it and only what the tables and charts
actually need is kept: the scores themselves, and the much smaller reporting grid for the
fan charts.

## What a model is allowed to see

A forecaster is handed ``panel[:, origin.train]`` and nothing else. It receives no dates,
no future values and no argument it could use to reach past the origin, which is a
stronger guarantee than a convention that the model should not look: there is nothing to
look at. :mod:`headroom.backtest.origins` fixes where the boundary is and
`tests/test_backtest.py` asserts it.
"""

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from headroom.backtest.origins import Origins
from headroom.hierarchy.spec import Hierarchy
from headroom.score import levels as levels_module
from headroom.score.coverage import covered, interval_bounds
from headroom.score.crps import crossing_rate, crps_from_quantiles, mae, sort_quantiles
from headroom.score.pinball import pinball_by_level
from headroom.score.width import width


class Forecaster(Protocol):
    """What the backtest requires of a model.

    Deliberately one method. A model that needed dates, or the hierarchy, or anything
    else about the world would be a model that could reach past its origin.
    """

    def forecast(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return quantile forecasts, shape ``(n_series, horizon, n_levels)``."""
        ...


@dataclass(frozen=True, slots=True)
class Scores:
    """Every score a backtest produces, indexed by origin, node and horizon step.

    Attributes:
        model: The model's name.
        hierarchy: The hierarchy the nodes belong to.
        origins: The schedule that produced these scores.
        crps: CRPS, shape ``(n_origins, n_nodes, horizon)``.
        pinball: Pinball loss on the reporting grid, shape
            ``(n_origins, n_nodes, horizon, n_reporting)``.
        absolute_error: Absolute error of the median. A footnote, never the headline.
        hits: Whether each central interval covered, shape
            ``(n_origins, n_nodes, horizon, n_nominal)``.
        widths: Width of each central interval, same shape as ``hits``.
        reporting_quantiles: Forecasts on the reporting grid, kept for the fan charts.
        nominal: The nominal coverages ``hits`` and ``widths`` are indexed by.
        crossing_rate: Share of forecasts whose raw quantiles crossed before sorting.
        fit_seconds: Wall-clock seconds spent inside the model.
    """

    model: str
    hierarchy: Hierarchy
    origins: Origins
    crps: npt.NDArray[np.float64]
    pinball: npt.NDArray[np.float64]
    absolute_error: npt.NDArray[np.float64]
    hits: npt.NDArray[np.bool_]
    widths: npt.NDArray[np.float64]
    reporting_quantiles: npt.NDArray[np.float64]
    nominal: tuple[float, ...]
    crossing_rate: float
    fit_seconds: float

    def by_origin(
        self, score: npt.NDArray[np.float64], level: str | None = None
    ) -> npt.NDArray[np.float64]:
        """Average a score over nodes and horizon, leaving one value per origin.

        This is the shape the block bootstrap takes: one number per origin, in origin
        order, so that the dependence between neighbouring origins is what gets resampled.

        Args:
            score: Any array shaped like :attr:`crps`.
            level: Restrict to one hierarchy level. All nodes if omitted.

        Returns:
            One value per origin, shape ``(n_origins,)``.
        """
        if level is None:
            return np.asarray(score.mean(axis=(1, 2)), dtype=np.float64)
        rows = self.hierarchy.rows_at(level)  # type: ignore[arg-type]
        return np.asarray(score[:, rows, :].mean(axis=(1, 2)), dtype=np.float64)

    def hits_by_day(
        self, nominal_index: int, level: str | None = None
    ) -> npt.NDArray[np.bool_]:
        """Flatten coverage into one series per target day, in calendar order.

        The rolling coverage chart needs coverage through *time*, not through origins,
        and a 14-day horizon means several origins forecast the same day. Each origin's
        horizon is laid out in order and the result is read as a stream of forecasts, so
        the window slides over the record the way a reader expects.

        Args:
            nominal_index: Which nominal coverage, indexing :attr:`nominal`.
            level: Restrict to one hierarchy level. All nodes if omitted.

        Returns:
            A one-dimensional boolean series in origin-then-horizon order.
        """
        hits = self.hits[..., nominal_index]
        if level is not None:
            rows = self.hierarchy.rows_at(level)  # type: ignore[arg-type]
            hits = np.take(hits, rows, axis=1)
        return np.asarray(hits.mean(axis=1).ravel() >= 0.5, dtype=np.bool_)


def run(
    model: Forecaster,
    name: str,
    panel_values: npt.NDArray[np.float64],
    hierarchy: Hierarchy,
    origins: Origins,
    scoring_levels: npt.NDArray[np.float64] | None = None,
    reporting_levels: npt.NDArray[np.float64] | None = None,
    nominal: tuple[float, ...] = levels_module.NOMINAL_COVERAGES,
) -> Scores:
    """Run a forecaster over every origin and return its scores.

    Args:
        model: The forecaster.
        name: Its name, for the tables.
        panel_values: Node values, shape ``(n_nodes, n_days)``.
        hierarchy: The hierarchy the nodes belong to.
        origins: The origin schedule.
        scoring_levels: Grid CRPS is integrated over. Defaults to the dense scoring grid.
        reporting_levels: Grid the tables and charts use. Defaults to the reporting grid.
        nominal: Nominal coverages to measure.

    Returns:
        The scores.

    Raises:
        ValueError: The panel does not match the hierarchy or the schedule.
    """
    scoring = levels_module.SCORING if scoring_levels is None else scoring_levels
    reporting = levels_module.REPORTING if reporting_levels is None else reporting_levels

    if panel_values.shape[0] != hierarchy.n_nodes:
        raise ValueError(
            f"panel has {panel_values.shape[0]} nodes, hierarchy has {hierarchy.n_nodes}"
        )
    if panel_values.shape[1] != len(origins.days):
        raise ValueError(
            f"panel has {panel_values.shape[1]} days, the schedule has {len(origins.days)}"
        )

    n_origins, n_nodes, horizon = len(origins), hierarchy.n_nodes, origins.horizon
    crps = np.empty((n_origins, n_nodes, horizon))
    pinball = np.empty((n_origins, n_nodes, horizon, reporting.size))
    absolute = np.empty((n_origins, n_nodes, horizon))
    hits = np.empty((n_origins, n_nodes, horizon, len(nominal)), dtype=np.bool_)
    widths = np.empty((n_origins, n_nodes, horizon, len(nominal)))
    kept = np.empty((n_origins, n_nodes, horizon, reporting.size))

    # Both grids are needed at every origin: the dense one to integrate CRPS, the sparse
    # one for the tables. Asking the model once for their union is cheaper than twice and
    # guarantees the two come from the same forecast.
    combined, scoring_at, reporting_at = _merge_grids(scoring, reporting)

    crossed = 0.0
    fit_seconds = 0.0
    for origin in origins:
        train = panel_values[:, origin.train]
        actual = panel_values[:, origin.target]

        started = time.perf_counter()
        raw = model.forecast(train, horizon, combined)
        fit_seconds += time.perf_counter() - started

        crossed += crossing_rate(raw)
        quantiles = sort_quantiles(raw)
        scoring_q = quantiles[..., scoring_at]
        reporting_q = quantiles[..., reporting_at]

        i = origin.number
        crps[i] = crps_from_quantiles(scoring_q, actual, scoring)
        pinball[i] = pinball_by_level(reporting_q, actual, reporting)
        absolute[i] = mae(reporting_q[..., reporting.size // 2], actual)
        kept[i] = reporting_q
        for j, coverage in enumerate(nominal):
            lower, upper = interval_bounds(reporting_q, reporting, coverage)
            hits[i, :, :, j] = covered(lower, upper, actual)
            widths[i, :, :, j] = width(reporting_q, reporting, coverage)

    return Scores(
        model=name,
        hierarchy=hierarchy,
        origins=origins,
        crps=crps,
        pinball=pinball,
        absolute_error=absolute,
        hits=hits,
        widths=widths,
        reporting_quantiles=kept,
        nominal=nominal,
        crossing_rate=crossed / n_origins,
        fit_seconds=fit_seconds,
    )


def _merge_grids(
    scoring: npt.NDArray[np.float64], reporting: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Merge two quantile grids into one, with an index back into each.

    Args:
        scoring: The dense grid CRPS is integrated over.
        reporting: The sparse grid the tables use.

    Returns:
        ``(combined, scoring_at, reporting_at)`` where the two index arrays select each
        original grid out of ``combined``.
    """
    combined = np.unique(np.concatenate([scoring, reporting]))
    return (
        combined,
        np.searchsorted(combined, scoring).astype(np.intp),
        np.searchsorted(combined, reporting).astype(np.intp),
    )
