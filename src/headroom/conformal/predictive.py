"""A predictive distribution for a model that forecasts only a median.

CRPS is scored over a 199-level quantile grid, and a model that forecasts one number per
day has no grid to score. LightGBM here (PLAN.md section 2.7b), and TimesFM later, forecast
a median, so their distribution is built from **their own out-of-sample errors at earlier
origins**: the forecast at an origin plus the order statistics of the signed errors the
same model made, at the same node and horizon step, over the previous :data:`WINDOW`
origins whose outcomes were already known.

This is a conformal predictive distribution, and it is built with the same machinery as
the conformal intervals in this package so that nothing about it is new:

* the same trailing window of 52 origins as :class:`headroom.conformal.split.SplitConformal`;
* the same feedback rule, :func:`headroom.conformal.split.available_upto`, so a forecast
  never calibrates on an error whose outcome had not happened yet;
* the same order-statistic convention as
  :func:`headroom.conformal.split.conformal_quantile`: the ``q`` quantile is the ``k``-th
  smallest error with ``k = ceil((n + 1) q)``, clipped to ``[1, n]``. Under
  exchangeability that puts at most ``q`` of the probability below the level and at most
  ``1 - q`` above it, at every level up to ``n / (n + 1)``. Above that, which with 52
  errors is the top of the 199-level grid, there are too few errors to certify the level
  and the widest one is used, as in :func:`~headroom.conformal.split.conformal_quantile`.

Two properties differ from the conformal intervals and are deliberate. The errors are
**signed**, so the distribution can be skewed the way emergency demand is. And an origin
gets a distribution **only once the full window is available** at every horizon step, so
every scored origin is calibrated on the same number of errors. That costs the first year
of origins, and any comparison with another model is made on the origins both have.

The assumption is the one :mod:`headroom.conformal.split` states: exchangeability, which a
time series does not have. The distribution's coverage is measured and reported, not
guaranteed, and it inherits the ceiling in :mod:`headroom.conformal.aci`: it cannot be
wider than the widest error in its window.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from headroom.conformal.split import WINDOW, available_upto
from headroom.score.levels import check_levels


@dataclass(frozen=True, slots=True)
class PredictiveQuantiles:
    """A predictive distribution over a backtest, for the origins that have one.

    Attributes:
        quantiles: Shape ``(n_origins, n_nodes, horizon, n_levels)``, ``nan`` for origins
            without a full calibration window.
        first_valid: The first origin with a distribution. Every later origin has one.
    """

    quantiles: npt.NDArray[np.float64]
    first_valid: int


def order_statistic_ranks(levels: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.intp]:
    """Return the zero-based rank of the sorted error each quantile level uses.

    Args:
        levels: Quantile levels, strictly inside ``(0, 1)``.
        n: Number of errors in the calibration window.

    Returns:
        ``ceil((n + 1) * q) - 1`` for each level, clipped to ``[0, n - 1]``.

    Raises:
        ValueError: ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"need at least one calibration error, got {n}")
    k = np.ceil((n + 1) * levels).astype(np.intp)
    return np.clip(k, 1, n) - 1


def first_valid_origin(horizon: int, origin_step: int, window: int = WINDOW) -> int:
    """Return the first origin at which every horizon step has a full window.

    The longest horizon step has the longest feedback lag, so it decides.

    Args:
        horizon: Days forecast ahead.
        origin_step: Days between origins.
        window: Calibration window in origins.

    Returns:
        The origin number.
    """
    origin = 0
    while available_upto(origin, horizon, origin_step) < window:
        origin += 1
    return origin


def predictive_quantiles(
    point: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
    origin_step: int,
    window: int = WINDOW,
) -> PredictiveQuantiles:
    """Build a predictive distribution from a point forecast's own past errors.

    Args:
        point: The model's forecasts, shape ``(n_origins, n_nodes, horizon)``.
        actual: What happened, same shape.
        levels: The quantile grid.
        origin_step: Days between origins, for the feedback rule.
        window: Calibration window in origins.

    Returns:
        The distribution, floored at zero because counts cannot be negative.

    Raises:
        ValueError: The shapes disagree, or the backtest is too short for any origin to
            have a full window.
    """
    check_levels(levels)
    if point.shape != actual.shape or point.ndim != 3:
        raise ValueError(f"point {point.shape} and actual {actual.shape} must match, 3-D")
    n_origins, n_nodes, horizon = point.shape
    first = first_valid_origin(horizon, origin_step, window)
    if first >= n_origins:
        raise ValueError(
            f"{n_origins} origins is too few: a {window}-origin window at a {horizon}-day "
            f"horizon first completes at origin {first}"
        )

    errors = actual - point
    ranks = order_statistic_ranks(levels, window)
    out = np.full((n_origins, n_nodes, horizon, levels.size), np.nan)
    for step in range(1, horizon + 1):
        for origin in range(first, n_origins):
            end = available_upto(origin, step, origin_step)
            calibration = np.sort(errors[end - window : end, :, step - 1], axis=0)
            offsets = calibration[ranks, :].T  # (n_nodes, n_levels)
            out[origin, :, step - 1, :] = point[origin, :, step - 1, np.newaxis] + offsets

    np.clip(out, 0.0, None, out=out)
    return PredictiveQuantiles(quantiles=out, first_valid=first)
