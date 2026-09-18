"""Apply a conformal method to a whole backtest.

Each ``(node, horizon step)`` pair is calibrated separately. Two reasons, and both would
be mistakes to skip.

A **14-day-ahead** forecast is more uncertain than a 1-day-ahead one, so pooling their
residuals would widen the short horizons and narrow the long ones, hiding the very thing
a planner needs to know about the horizon they are planning over.

A **dispatch area** averaging 20 arrivals a day and the **city** averaging 3,818 are on
scales two orders of magnitude apart. A shared calibration set would be dominated by the
city and would make every leaf interval absurd.

The cost is that each pair calibrates on its own window of origins rather than borrowing
strength across the hierarchy, which is why the window is a year rather than a month.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from headroom.backtest.run import Scores
from headroom.conformal.split import nonconformity
from headroom.hierarchy.spec import Hierarchy
from headroom.score.coverage import covered


class ConformalMethod(Protocol):
    """What :func:`apply` requires of a conformal method."""

    @property
    def name(self) -> str:
        """A label for the tables."""
        ...

    def widths(
        self, scores: npt.NDArray[np.float64], horizon_step: int, origin_step: int
    ) -> npt.NDArray[np.float64]:
        """Return a half-width per origin for one series at one horizon step."""
        ...


@dataclass(frozen=True, slots=True)
class ConformalScores:
    """A conformal method's intervals over a backtest, and how they did.

    Attributes:
        method: The method's name.
        base_model: The model whose point forecast was wrapped.
        hierarchy: The hierarchy the nodes belong to.
        alpha: Target miscoverage, so nominal coverage is ``1 - alpha``.
        lower: Interval lower bounds, shape ``(n_origins, n_nodes, horizon)``.
        upper: Interval upper bounds, same shape.
        hits: Whether each interval covered, same shape.
        widths: Full interval widths, same shape.
        valid: Whether an interval was produced at all. False before a node and horizon
            step has observed enough history to calibrate on.
    """

    method: str
    base_model: str
    hierarchy: Hierarchy
    alpha: float
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    hits: npt.NDArray[np.bool_]
    widths: npt.NDArray[np.float64]
    valid: npt.NDArray[np.bool_]

    def coverage(self, level: str | None = None) -> float:
        """Empirical coverage over every valid interval.

        Args:
            level: Restrict to one hierarchy level. All nodes if omitted.

        Returns:
            The share covered, between 0 and 1.
        """
        hits = np.asarray(self._select(self.hits, level), dtype=np.bool_)
        valid = np.asarray(self._select(self.valid, level), dtype=np.bool_)
        return float(hits[valid].mean())

    def mean_width(self, level: str | None = None) -> float:
        """Mean interval width over every valid interval.

        Args:
            level: Restrict to one hierarchy level. All nodes if omitted.

        Returns:
            The mean width, in the units of the data.
        """
        widths = np.asarray(self._select(self.widths, level), dtype=np.float64)
        valid = np.asarray(self._select(self.valid, level), dtype=np.bool_)
        return float(widths[valid].mean())

    def coverage_by_origin(self, level: str | None = None) -> npt.NDArray[np.float64]:
        """Coverage at each origin, for the rolling chart and the bootstrap.

        Args:
            level: Restrict to one hierarchy level. All nodes if omitted.

        Returns:
            One coverage value per origin, ``nan`` where no interval was valid.
        """
        hits = np.asarray(self._select(self.hits, level), dtype=np.bool_)
        valid = np.asarray(self._select(self.valid, level), dtype=np.bool_)
        covered_count = (hits & valid).sum(axis=(1, 2)).astype(np.float64)
        total = valid.sum(axis=(1, 2)).astype(np.float64)
        with np.errstate(invalid="ignore"):
            return np.where(total > 0, covered_count / np.maximum(total, 1.0), np.nan)

    def width_by_origin(self, level: str | None = None) -> npt.NDArray[np.float64]:
        """Mean interval width at each origin, for the chart that sits under coverage.

        Coverage alone cannot be read: a method can reach nominal by being wide enough to
        cover anything. The two are plotted together for that reason.

        Args:
            level: Restrict to one hierarchy level. All nodes if omitted.

        Returns:
            One mean width per origin, ``nan`` where no interval was valid.
        """
        widths = np.asarray(self._select(self.widths, level), dtype=np.float64)
        valid = np.asarray(self._select(self.valid, level), dtype=np.bool_)
        total = valid.sum(axis=(1, 2)).astype(np.float64)
        summed = np.where(valid, np.nan_to_num(widths), 0.0).sum(axis=(1, 2))
        with np.errstate(invalid="ignore"):
            return np.where(total > 0, summed / np.maximum(total, 1.0), np.nan)

    def _select(
        self, values: npt.NDArray[np.generic], level: str | None
    ) -> npt.NDArray[np.generic]:
        """Restrict an array to one hierarchy level.

        Args:
            values: Any array shaped like :attr:`hits`.
            level: The level, or None for all nodes.

        Returns:
            The array, possibly with fewer nodes.
        """
        if level is None:
            return values
        rows = self.hierarchy.rows_at(level)  # type: ignore[arg-type]
        return np.take(values, rows, axis=1)


def apply(
    method: ConformalMethod,
    scores: Scores,
    panel_values: npt.NDArray[np.float64],
    alpha: float,
) -> ConformalScores:
    """Wrap a backtest's point forecasts in conformal intervals.

    Args:
        method: The conformal method.
        scores: A completed backtest, whose reporting quantiles supply the point forecast.
        panel_values: The panel the backtest ran on, shape ``(n_nodes, n_days)``.
        alpha: Target miscoverage, so nominal coverage is ``1 - alpha``.

    Returns:
        The intervals and how they did.

    Raises:
        ValueError: The panel does not match the backtest.
    """
    origins = scores.origins
    n_origins, n_nodes, horizon = scores.crps.shape
    if panel_values.shape[0] != n_nodes:
        raise ValueError(f"panel has {panel_values.shape[0]} nodes, backtest has {n_nodes}")

    median_at = scores.reporting_quantiles.shape[-1] // 2
    point = scores.reporting_quantiles[..., median_at]

    actual = np.empty((n_origins, n_nodes, horizon))
    for origin in origins:
        actual[origin.number] = panel_values[:, origin.target]

    residuals = nonconformity(actual, point)

    lower = np.full(residuals.shape, np.nan)
    upper = np.full(residuals.shape, np.nan)
    half = np.full(residuals.shape, np.nan)
    for node in range(n_nodes):
        for step in range(horizon):
            half[:, node, step] = method.widths(
                residuals[:, node, step], step + 1, origins.step
            )

    valid = np.isfinite(half)
    with np.errstate(invalid="ignore"):
        lower = np.maximum(point - half, 0.0)  # counts cannot be negative
        upper = point + half

    hits = np.zeros(residuals.shape, dtype=np.bool_)
    hits[valid] = covered(lower[valid], upper[valid], actual[valid])

    return ConformalScores(
        method=method.name,
        base_model=scores.model,
        hierarchy=scores.hierarchy,
        alpha=alpha,
        lower=lower,
        upper=upper,
        hits=hits,
        widths=np.where(valid, upper - lower, np.nan),
        valid=valid,
    )
