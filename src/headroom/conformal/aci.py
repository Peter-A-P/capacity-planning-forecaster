"""Adaptive conformal inference: the miscoverage level itself is updated online.

Gibbs and Candes (2021). The construction is one line, and the line is the whole idea:

    alpha_next = alpha_now + gamma * (target - missed)

where ``missed`` is 1 if the last observed interval failed to contain its outcome and 0 if
it contained it. Miss more often than the target and ``alpha_now`` falls, which widens the
next interval; miss less often and it rises, which tightens it. The interval is otherwise
built exactly as in split conformal, from the same rolling calibration window, so the only
difference between the two methods in this repository is whether this line runs.

## What it guarantees, and what it does not

**It guarantees long-run average coverage.** Because ``alpha`` is bounded in its updates,
the realised miscoverage rate over ``T`` steps converges to the target at rate ``O(1/T)``,
**for any data sequence at all**: no exchangeability, no stationarity, no assumption about
the model. An adversary choosing the data cannot break it. That is why it is the right
tool for a series with a March 2020 in it.

**It does not guarantee per-period coverage.** Over any particular window it can be far
from nominal, and it is guaranteed only to come back. It learns from misses, so it can only
respond to a shift *after* the shift has cost it some coverage. The lag is roughly
``1 / gamma`` steps.

This distinction is the single most likely way for this project to be wrong in public, so
it is stated in `docs/methods.md`, in the README beside the coverage chart, and here.
Reporting "coverage held through the shift" without it would be a claim the method does
not make.

## The ceiling this construction cannot pass

Measured on the NYC panel, and worth knowing before trusting the method: **driving
``alpha`` to zero buys the largest nonconformity score in the calibration window and not
one unit more.** At the city through March 2020 the widest interval the window could ever
produce was 1,374, the method reached 1,262, and the worst residual of the shift was
1,574. No value of ``alpha`` could have covered that day. The adaptation was not too slow;
it ran out of room.

That is a property of re-quantiling inside a bounded calibration set, not of the tuning.
`docs/methods.md` has the table and names the standard remedy, which is to make the score
scale-free so the interval can exceed anything the window has literally seen. Nothing here
claims that yet.

## Choosing gamma

Large ``gamma`` reacts fast and is noisy: it chases individual misses and the interval
width oscillates. Small ``gamma`` is stable and slow, and a shift costs more coverage
before it responds. There is no value that is right for both, which is the honest reason
:mod:`headroom.conformal.agaci` exists: it removes the choice by running several and
aggregating them.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from headroom.conformal.split import WINDOW, available_upto, conformal_quantile

#: Step size when a single one is used. 0.01 responds over roughly a hundred origins,
#: about two years at a weekly step, which is slow. `docs/methods.md` reports what the
#: choice costs; :class:`headroom.conformal.agaci.AggregatedConformal` avoids making it.
GAMMA: Final[float] = 0.01

#: The candidate step sizes AgACI aggregates over, spanning three orders of magnitude.
GAMMA_GRID: Final[tuple[float, ...]] = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)

#: ``alpha`` is clipped into this range. At 0 the interval would be the widest score seen
#: and never adapt down; at 1 it would be zero-width and never adapt up. Clipping keeps
#: the update reversible, which is what the convergence argument needs.
ALPHA_FLOOR: Final[float] = 1e-4
ALPHA_CEILING: Final[float] = 1.0 - 1e-4


@dataclass(frozen=True, slots=True)
class AdaptiveConformal:
    """Adaptive conformal inference with a single step size.

    Attributes:
        alpha: The target miscoverage the update aims at.
        gamma: Step size of the online update.
        window: Calibration window in origins.
    """

    alpha: float
    gamma: float = GAMMA
    window: int = WINDOW

    @property
    def name(self) -> str:
        """A label for the tables."""
        return f"adaptive conformal (alpha={self.alpha:.2f}, gamma={self.gamma:g})"

    def widths(
        self, scores: npt.NDArray[np.float64], horizon_step: int, origin_step: int
    ) -> npt.NDArray[np.float64]:
        """Return a half-width per origin for one series at one horizon step.

        Args:
            scores: Nonconformity scores in origin order, shape ``(n_origins,)``.
            horizon_step: Days ahead, counting from 1.
            origin_step: Days between origins.

        Returns:
            Half-widths, shape ``(n_origins,)``, ``nan`` before anything is observed.
        """
        return self.trace(scores, horizon_step, origin_step)[0]

    def trace(
        self, scores: npt.NDArray[np.float64], horizon_step: int, origin_step: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the half-widths and the path ``alpha`` took.

        The alpha path is what the diagnostic chart plots: it shows the method noticing
        the shift, and how long it took.

        Args:
            scores: Nonconformity scores in origin order.
            horizon_step: Days ahead, counting from 1.
            origin_step: Days between origins.

        Returns:
            ``(widths, alphas)``, each shape ``(n_origins,)``.
        """
        n = scores.size
        widths = np.full(n, np.nan)
        alphas = np.full(n, np.nan)

        current = self.alpha
        consumed = 0  # how many observed outcomes have already fed the update

        for origin in range(n):
            end = available_upto(origin, horizon_step, origin_step)

            # Feed the update every outcome that has become visible since the last origin,
            # in order. Skipping any would let the method drift; using one that has not
            # happened yet would be look-ahead.
            while consumed < end:
                calibration = scores[max(0, consumed - self.window) : consumed]
                if calibration.size > 0:
                    half_width = conformal_quantile(calibration, _clip(current))
                    missed = float(scores[consumed] > half_width)
                    current = _clip(current + self.gamma * (self.alpha - missed))
                consumed += 1

            if end == 0:
                continue
            alphas[origin] = current
            widths[origin] = conformal_quantile(
                scores[max(0, end - self.window) : end], _clip(current)
            )

        return widths, alphas


def _clip(alpha: float) -> float:
    """Keep ``alpha`` inside the range where the update stays reversible.

    Args:
        alpha: The proposed miscoverage level.

    Returns:
        The value clipped into ``[ALPHA_FLOOR, ALPHA_CEILING]``.
    """
    return float(min(max(alpha, ALPHA_FLOOR), ALPHA_CEILING))
