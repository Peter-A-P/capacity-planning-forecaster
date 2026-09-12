"""Split conformal, and the feedback rule every method here obeys.

Split conformal takes the absolute residuals of a base forecast on a calibration set and
uses their ``1 - alpha`` quantile as a half-width. Under **exchangeability** that gives
finite-sample marginal coverage of at least ``1 - alpha``, with no assumption about the
model or the distribution. It is a genuinely remarkable guarantee.

It does not apply here. A time series is not exchangeable, and the whole point of this
project is the period when it is least exchangeable of all. Split conformal is kept
because watching it fail is the evidence (PLAN.md section 9, Rule C candidate 1), not
because its guarantee is believed.

## The one thing that must not be got wrong

A conformal method learns from its own past errors, so it is the part of this project
most able to cheat without anyone noticing. At an origin, the outcome of a forecast made
``h`` days ahead is known only ``h`` days later. With origins seven days apart and a
horizon of fourteen, the error of a 14-day-ahead forecast made at the previous origin has
**not** happened yet when the next forecast is made.

:func:`available_upto` is the only place that rule lives. Every method in this package
calibrates on ``scores[: available_upto(...)]`` and nothing else, and
`tests/test_conformal.py` asserts that a method given future scores would have produced a
different answer, so the guard cannot rot into a comment.

## The rolling calibration window

The classical construction holds one fixed calibration set aside forever. That would make
the comparison against adaptive conformal unfair in the adaptive method's favour, because
two things would differ at once: the calibration data and the miscoverage level.

So every method here uses the **same** trailing window of :data:`WINDOW` origins, and the
only difference between split and adaptive conformal is whether ``alpha`` moves. The
comparison is then about adaptation and nothing else.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: Calibration window, in origins. 52 at a 7-day step is one year, which covers a whole
#: annual cycle so the residuals are not drawn from one season only.
WINDOW: Final[int] = 52


def horizon_lag(horizon_step: int, origin_step: int) -> int:
    """Return how many origins pass before a forecast's outcome is known.

    Args:
        horizon_step: Days ahead the forecast was made, counting from 1.
        origin_step: Days between origins.

    Returns:
        Origins of delay, at least 1.

    Raises:
        ValueError: Either argument is not positive.
    """
    if horizon_step < 1 or origin_step < 1:
        raise ValueError(f"positive arguments required, got {horizon_step}, {origin_step}")
    return max(1, -(-horizon_step // origin_step))  # ceil


def available_upto(origin: int, horizon_step: int, origin_step: int) -> int:
    """Return how many past origins' scores are known when forecasting at an origin.

    The answer is an exclusive end index into a per-origin score array, so a caller writes
    ``scores[:available_upto(...)]`` and cannot accidentally include the present.

    Args:
        origin: The origin's number, counting from zero.
        horizon_step: Days ahead being forecast, counting from 1.
        origin_step: Days between origins.

    Returns:
        The number of leading origins whose outcome at this horizon has been observed.
        Zero when nothing has been observed yet.
    """
    return max(0, origin - horizon_lag(horizon_step, origin_step) + 1)


def conformal_quantile(scores: npt.NDArray[np.float64], alpha: float) -> float:
    """Return the conformal quantile of a set of nonconformity scores.

    The answer is the ``k``-th smallest score with ``k = ceil((n + 1) * (1 - alpha))``.
    The ``n + 1`` is the finite-sample correction, and it is what makes the exchangeable
    guarantee hold exactly at small ``n`` rather than approximately; dropping it is a
    common and quiet way to undercover.

    It is deliberately an **order statistic** and not a call to a quantile function. The
    usual quantile convention interpolates over ``n - 1`` intervals, which lands one order
    statistic away from what the conformal construction asks for and makes every interval
    slightly too wide. That error is invisible in a coverage table, because over-covering
    looks like caution rather than like a bug.

    When ``k`` exceeds ``n``, there are too few scores to certify the requested coverage
    and the widest observed score is returned. That is the honest answer, and it is why a
    method with a short calibration window starts conservative.

    Args:
        scores: Nonconformity scores, one per calibration point. Must be non-empty.
        alpha: Target miscoverage, strictly between 0 and 1.

    Returns:
        The half-width.

    Raises:
        ValueError: There are no scores, or alpha is out of range.
    """
    if scores.size == 0:
        raise ValueError("no calibration scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(np.max(scores))
    return float(np.partition(scores, k - 1)[k - 1])


def nonconformity(
    actual: npt.NDArray[np.float64], point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Absolute residual of a point forecast, the score every method here uses.

    Symmetric, so the interval is symmetric about the point forecast. Emergency demand is
    right-skewed and an asymmetric score would fit it better, but it would also change two
    things at once between this project's methods and its baseline. It is named in
    `PLAN.md` section 11 as deferred.

    Args:
        actual: Realised values.
        point: The base forecast's point prediction, usually its median.

    Returns:
        ``|actual - point|``.
    """
    return np.abs(actual - point)


@dataclass(frozen=True, slots=True)
class SplitConformal:
    """Split conformal with a fixed miscoverage level and a rolling calibration window.

    Attributes:
        alpha: Target miscoverage. Never moves; that is the whole point of the comparison.
        window: Calibration window in origins.
    """

    alpha: float
    window: int = WINDOW

    @property
    def name(self) -> str:
        """A label for the tables."""
        return f"split conformal (alpha={self.alpha:.2f})"

    def widths(
        self, scores: npt.NDArray[np.float64], horizon_step: int, origin_step: int
    ) -> npt.NDArray[np.float64]:
        """Return a half-width per origin for one series at one horizon step.

        Args:
            scores: Nonconformity scores in origin order, shape ``(n_origins,)``.
            horizon_step: Days ahead, counting from 1.
            origin_step: Days between origins.

        Returns:
            Half-widths, shape ``(n_origins,)``. Origins with no observed history yet get
            ``nan``, so they are excluded from coverage rather than counted as a miss.
        """
        out = np.full(scores.size, np.nan)
        for origin in range(scores.size):
            end = available_upto(origin, horizon_step, origin_step)
            if end == 0:
                continue
            calibration = scores[max(0, end - self.window) : end]
            out[origin] = conformal_quantile(calibration, self.alpha)
        return out
