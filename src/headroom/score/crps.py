"""CRPS, the headline score.

CRPS is the whole predictive distribution's score against the single value that happened.
It is in the units of the data, it reduces to absolute error when the forecast is a point,
and unlike coverage it cannot be gamed by making the interval wider: a distribution that
buys coverage with width is charged for the width.

## Why this is a quadrature, and what that costs

There is no closed form for the CRPS of a distribution given only as a set of quantiles.
What there is, is the identity

    CRPS(F, y) = 2 * integral over tau in (0, 1) of pinball_tau(F_inverse(tau), y)

so the CRPS of a quantile forecast is twice the area under its own pinball losses. That
ties the two headline scores of PLAN.md section 1 together: the pinball table and the CRPS
column are the same numbers, integrated or not.

The integral is evaluated by the trapezoidal rule between the outermost quantile levels,
plus an exact term for each tail.

## The tails are not zero, and assuming they were would flatter the forecast

The obvious implementation pads the integrand with zero at tau = 0 and tau = 1. That is
wrong, and wrong in the direction that flatters a bad forecast. The integrand vanishes at
tau = 1 only when the observation lies strictly below the top of the forecast's support.
When the observation lands **outside** the quantile grid, which is precisely what a
distribution shift does, the integrand at that end is proportional to the size of the miss
and padding it to zero discards part of the penalty. The effect is not enormous, about half
a percent of the score on this grid, but it is systematic, it grows with the miss, and it
applies only to the forecasts that missed. A bias that lets through exactly the errors the
project exists to measure is not one to leave in.

So each tail is integrated in closed form instead, under the explicit assumption that the
distribution is flat beyond the outermost levels: ``q(tau) = q_first`` below the first
level and ``q(tau) = q_last`` above the last. Under that assumption the tail integrals are
exact, and the assumption is conservative rather than flattering, because a real
distribution with a longer tail puts its quantiles further from the observation and scores
better. Anything this reports is therefore an upper bound on what a heavier-tailed reading
of the same forecast would give. `docs/methods.md` states it beside the numbers.

The remaining error is discretisation error on the interior, it is always **downward**
(the integrand is concave in tau near its peak, so a chord runs below the curve), and it
depends only on how dense the grid is, not on the model being scored. Two consequences,
both of which `docs/methods.md` states:

* Every method is scored on the same grid, so the bias is common to all of them and the
  skill ratios that the tables actually report are almost unaffected by it.
* The absolute CRPS values are reported on :data:`headroom.score.levels.SCORING`, not on
  the reporting grid, because the reporting grid is chosen to be readable and is far too
  coarse to integrate against.

:func:`crps_normal` is the closed form for a normal, which is what the discretisation
error is measured against in the tests and in `docs/methods.md`.
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from headroom.score.pinball import pinball_by_level


def crps_from_quantiles(
    quantiles: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """CRPS of a quantile forecast, by integrating its pinball losses.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``, with the level axis
            last. They must be non-decreasing along that axis; see
            :func:`sort_quantiles` if a model can cross them.
        observed: Realised values, shape ``(...)``.
        levels: The quantile levels, strictly increasing inside ``(0, 1)``. Use
            :data:`headroom.score.levels.SCORING` unless there is a reason not to.

    Returns:
        CRPS, shape ``(...)``, in the units of the data.

    Raises:
        ValueError: The levels are unusable or do not match the last axis.
    """
    losses = pinball_by_level(quantiles, observed, levels)
    interior = np.asarray(np.trapezoid(losses, x=levels, axis=-1), dtype=np.float64)
    total: npt.NDArray[np.float64] = 2.0 * (
        interior
        + _lower_tail(quantiles, observed, levels)
        + _upper_tail(quantiles, observed, levels)
    )
    return total


def _lower_tail(
    quantiles: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Integrate the pinball loss over ``(0, levels[0])`` with a flat tail.

    With ``q(tau) = q_first`` throughout the tail, the integrand is ``tau * (y - q)`` when
    the observation is above ``q_first`` and ``(1 - tau) * (q - y)`` when it is below, and
    both integrate exactly.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.
        observed: Realised values, shape ``(...)``.
        levels: The quantile grid.

    Returns:
        The tail's contribution to the integral, shape ``(...)``.
    """
    first_level = float(levels[0])
    gap = observed - quantiles[..., 0]
    above = first_level**2 / 2.0 * gap
    below = (first_level - first_level**2 / 2.0) * (-gap)
    return np.where(gap >= 0.0, above, below)


def _upper_tail(
    quantiles: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Integrate the pinball loss over ``(levels[-1], 1)`` with a flat tail.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.
        observed: Realised values, shape ``(...)``.
        levels: The quantile grid.

    Returns:
        The tail's contribution to the integral, shape ``(...)``.
    """
    last_level = float(levels[-1])
    span = 1.0 - last_level
    gap = observed - quantiles[..., -1]
    # Integral of tau over (last, 1) is (1 - last^2) / 2; of (1 - tau) is span^2 / 2.
    above = (1.0 - last_level**2) / 2.0 * gap
    below = span**2 / 2.0 * (-gap)
    return np.where(gap >= 0.0, above, below)


def crps_normal(
    mean: npt.NDArray[np.float64] | float,
    sd: npt.NDArray[np.float64] | float,
    observed: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """CRPS of a normal forecast, in closed form.

    This exists to check :func:`crps_from_quantiles`, which is a quadrature and has no
    closed form of its own. It is not used to score anything: nothing in this project
    forecasts a normal.

    The identity is ``sd * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / sqrt(pi))`` with
    ``z = (observed - mean) / sd``.

    Args:
        mean: Predictive mean.
        sd: Predictive standard deviation, strictly positive.
        observed: Realised value.

    Returns:
        CRPS, broadcast over the inputs.

    Raises:
        ValueError: A standard deviation is not strictly positive.
    """
    sd_arr = np.asarray(sd, dtype=np.float64)
    if np.any(sd_arr <= 0.0):
        raise ValueError("sd must be strictly positive")
    z = (np.asarray(observed, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / sd_arr
    result: npt.NDArray[np.float64] = sd_arr * (
        z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    return result


def sort_quantiles(quantiles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Repair crossed quantiles by sorting along the level axis.

    Quantiles predicted independently per level can cross: a model can put its 0.6
    quantile below its 0.5. Sorting is the standard repair and it never makes any pinball
    loss worse, but it is a repair, so the backtest records how often it changed anything
    rather than applying it silently.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.

    Returns:
        The same array with the last axis sorted ascending.
    """
    return np.sort(quantiles, axis=-1)


def crossing_rate(quantiles: npt.NDArray[np.float64]) -> float:
    """Return the share of forecasts whose quantiles cross.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.

    Returns:
        The share of the leading axes' entries with at least one crossing, between 0
        and 1.
    """
    crossed = np.any(np.diff(quantiles, axis=-1) < 0.0, axis=-1)
    return float(np.mean(crossed))


def mae(
    point: npt.NDArray[np.float64], observed: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Mean absolute error, reported as a footnote and never as the headline.

    It is here because readers look for it, and because PLAN.md section 9's second
    candidate is what choosing a model by this number would have done. A point forecast
    cannot be scored on whether its range held.

    Args:
        point: The point forecast, usually the median.
        observed: Realised values.

    Returns:
        Absolute error, elementwise.
    """
    return np.abs(observed - point)
