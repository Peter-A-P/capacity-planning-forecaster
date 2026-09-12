"""The quantile grid every model predicts on.

One grid, fixed here, used by every model and every score. Two things depend on it.

**The reporting grid** is what the tables show: the central intervals a planner reads,
plus the tails a service level is chosen from. It is deliberately small, because each
level is a column in a results table and a row a reader has to follow.

**The scoring grid** is what CRPS is integrated over. CRPS from quantiles is a
quadrature, so its accuracy is set by how dense the grid is, and a grid chosen for
legibility is far too coarse for it. `docs/methods.md` reports the discretisation error
of both grids against the closed form.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

#: Levels the tables report. Symmetric, so every one of them pairs into a central
#: interval: 50, 80, 90, 95 and 98 percent.
REPORTING: Final[npt.NDArray[np.float64]] = np.array(
    [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
)

#: Levels CRPS is integrated over. 199 equally spaced levels from 0.005 to 0.995; see
#: :func:`headroom.score.crps.crps_from_quantiles` for why the grid has to be dense and
#: what it costs to make it denser.
SCORING: Final[npt.NDArray[np.float64]] = np.linspace(0.005, 0.995, 199)

#: Nominal coverages the coverage chart plots, as PLAN.md section 1 requires.
NOMINAL_COVERAGES: Final[tuple[float, ...]] = (0.80, 0.90, 0.95)


def interval_levels(coverage: float) -> tuple[float, float]:
    """Return the two quantile levels bounding a central interval.

    Args:
        coverage: Nominal coverage, strictly between 0 and 1.

    Returns:
        The lower and upper levels, for example ``(0.05, 0.95)`` for 0.90.

    Raises:
        ValueError: The coverage is not strictly between 0 and 1.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    tail = (1.0 - coverage) / 2.0
    return tail, 1.0 - tail


def check_levels(levels: npt.NDArray[np.float64]) -> None:
    """Check a quantile grid is usable as one.

    Args:
        levels: The quantile levels.

    Raises:
        ValueError: The levels are not strictly increasing inside ``(0, 1)``.
    """
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("levels must be a non-empty one-dimensional array")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be strictly increasing")
    if levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError(
            f"levels must lie strictly inside (0, 1), got {levels[0]}..{levels[-1]}"
        )
