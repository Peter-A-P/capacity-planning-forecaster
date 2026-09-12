"""Interval width, reported beside every coverage number.

Coverage bought by width is not a result (PLAN.md section 1). An interval wide enough to
cover everything covers everything; the question is what it cost to be right. CRPS already
charges for width, which is why it is the headline, but a planner reads the interval, so
the width is reported in the units they read it in.
"""

import numpy as np
import numpy.typing as npt

from headroom.score.coverage import interval_bounds


def width(
    quantiles: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
    coverage: float,
) -> npt.NDArray[np.float64]:
    """Width of a central interval.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.
        levels: The quantile grid.
        coverage: Nominal coverage, for example 0.90.

    Returns:
        The width, shape ``(...)``, in the units of the data.
    """
    lower, upper = interval_bounds(quantiles, levels, coverage)
    return upper - lower


def relative_width(
    quantiles: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
    coverage: float,
    observed: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interval width as a share of the level of the series.

    A width of 400 means one thing at the city, which averages 3,818 arrivals a day, and
    something else entirely at a dispatch area averaging 20. Dividing by the observation
    is what makes the widths comparable down the hierarchy, which the results tables need.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``.
        levels: The quantile grid.
        coverage: Nominal coverage.
        observed: Realised values, shape ``(...)``, strictly positive.

    Returns:
        Width divided by the observation, shape ``(...)``.

    Raises:
        ValueError: An observation is zero or negative, which would make the ratio
            meaningless rather than merely large.
    """
    if np.any(observed <= 0.0):
        raise ValueError(
            "relative width needs strictly positive observations; a leaf with zero "
            "arrivals has no scale to divide by, so report absolute width there"
        )
    return width(quantiles, levels, coverage) / observed
