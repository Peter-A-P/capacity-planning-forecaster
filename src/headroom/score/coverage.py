"""Empirical coverage, and coverage through time.

Coverage on its own is not a result. An interval from minus infinity to plus infinity
covers everything, so :mod:`headroom.score.width` is reported beside every number here,
and PLAN.md section 1 says so in as many words.

The rolling window is the point of this module. A single coverage figure over a whole
backtest can sit exactly on nominal while the interval was far too narrow for six weeks
in 2020 and slightly too wide for the eighteen years around it. The headline chart plots
coverage in a rolling window through the March 2020 shift precisely so that averaging
cannot hide it.
"""

import numpy as np
import numpy.typing as npt

from headroom.score.levels import check_levels, interval_levels

#: Width of the rolling coverage window, in origins. Ninety days is long enough that the
#: estimate is not noise (at 90 percent nominal its standard error is about 3 points) and
#: short enough to resolve a shift that took three weeks. PLAN.md section 1 fixes it.
ROLLING_WINDOW: int = 90


def level_index(levels: npt.NDArray[np.float64], level: float) -> int:
    """Return the position of a level in the grid.

    Interval bounds are looked up, never interpolated. Interpolating between two
    quantiles would put an approximation inside the coverage number, which is the one
    number in this project that has to be exactly what it says it is.

    Args:
        levels: The quantile grid.
        level: The level to find.

    Returns:
        Its index in ``levels``.

    Raises:
        ValueError: The level is not in the grid.
    """
    check_levels(levels)
    hits = np.flatnonzero(np.isclose(levels, level, rtol=0.0, atol=1e-12))
    if hits.size != 1:
        raise ValueError(
            f"level {level} is not in the quantile grid; coverage bounds are looked up "
            f"rather than interpolated, so the grid must contain it. Grid: {levels}"
        )
    return int(hits[0])


def interval_bounds(
    quantiles: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
    coverage: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the lower and upper bounds of a central interval.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.
        levels: The quantile grid.
        coverage: Nominal coverage, for example 0.90.

    Returns:
        ``(lower, upper)``, each shape ``(...)``.

    Raises:
        ValueError: The grid does not contain both bounding levels.
    """
    lo_level, hi_level = interval_levels(coverage)
    return (
        quantiles[..., level_index(levels, lo_level)],
        quantiles[..., level_index(levels, hi_level)],
    )


def covered(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Report, elementwise, whether the interval contained the observation.

    The interval is closed. Daily counts are integers and a forecast quantile that lands
    exactly on one is common enough that treating the endpoint as a miss would bias
    coverage downward for no reason.

    Args:
        lower: Lower bounds.
        upper: Upper bounds.
        observed: Realised values.

    Returns:
        A boolean array of the broadcast shape.
    """
    return (observed >= lower) & (observed <= upper)


def empirical_coverage(hits: npt.NDArray[np.bool_]) -> float:
    """Return the share of observations the interval contained.

    Args:
        hits: Output of :func:`covered`.

    Returns:
        The share, between 0 and 1.

    Raises:
        ValueError: There is nothing to average.
    """
    if hits.size == 0:
        raise ValueError("no observations to take coverage over")
    return float(np.mean(hits))


def rolling_coverage(
    hits: npt.NDArray[np.bool_], window: int = ROLLING_WINDOW
) -> npt.NDArray[np.float64]:
    """Coverage in a trailing window, one value per position.

    The window is trailing, not centred. A centred window would let an origin's coverage
    be computed partly from days after it, which is exactly the look-ahead the whole
    backtest is built to avoid, and it would smear the March 2020 break backwards across
    six weeks that had not seen it yet.

    Positions before the window is full are ``nan`` rather than a short-window average,
    so the chart cannot show a noisy estimate as if it were a settled one.

    Args:
        hits: Output of :func:`covered`, one entry per origin, in origin order.
        window: Window length in origins.

    Returns:
        Rolling coverage, same length as ``hits``, ``nan`` for the first
        ``window - 1`` positions.

    Raises:
        ValueError: The window is not a positive integer, or is longer than the series.
    """
    if hits.ndim != 1:
        raise ValueError("rolling coverage takes one series at a time")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if window > hits.size:
        raise ValueError(f"window of {window} is longer than the {hits.size} origins")

    cumulative = np.concatenate(([0.0], np.cumsum(hits.astype(np.float64))))
    totals = cumulative[window:] - cumulative[:-window]
    out = np.full(hits.size, np.nan)
    out[window - 1 :] = totals / window
    return out


def worst_window(
    hits: npt.NDArray[np.bool_], window: int = ROLLING_WINDOW
) -> tuple[int, float]:
    """Return the position and value of the lowest rolling coverage.

    This is the number the README reports beside whole-period coverage, because it is the
    one a planner would have been hurt by: it is the worst the interval ever got.

    Args:
        hits: Output of :func:`covered`, in origin order.
        window: Window length in origins.

    Returns:
        ``(index, coverage)`` for the trailing window ending at ``index``.
    """
    rolling = rolling_coverage(hits, window)
    index = int(np.nanargmin(rolling))
    return index, float(rolling[index])
