"""Aggregated adaptive conformal: run several step sizes and let the data pick.

Zaffran and others (2022). :mod:`headroom.conformal.aci` works, but it replaces one
arbitrary choice (a fixed interval width) with another (``gamma``), and the right
``gamma`` is not knowable in advance: it depends on how abruptly the series shifts, which
is the thing being forecast. Reporting a result at a ``gamma`` chosen after seeing which
one did best on the 2020 shift would not be a result at all.

So every candidate step size runs as an **expert**, each producing its own half-width at
each origin, and the reported width is a weighted average of theirs. Weights are updated
online from each expert's realised pinball loss, so an expert that has been performing
well recently carries more of the answer. Nothing is chosen with hindsight: the weight an
expert has at an origin depends only on losses already observed at that origin, through
the same :func:`headroom.conformal.split.available_upto` rule as everything else.

## The aggregation rule

Exponentially weighted averaging. Expert ``k`` carries weight proportional to
``exp(-learning_rate * cumulative_loss_k)``, so weights move smoothly and no expert is
ever discarded outright, which matters because the expert that is right during a shift is
usually the one that was wrong before it.

The loss is the **pinball loss of the interval bound**, not squared error. It is the loss
whose minimiser is the quantile the method is trying to produce, so an expert is rewarded
for the thing it is for: an expert whose interval is too wide is penalised as well as one
whose interval is too narrow.

The half-width is aggregated on the **upper bound**, with the lower bound mirrored, since
the nonconformity score is symmetric.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from headroom.conformal.aci import GAMMA_GRID, AdaptiveConformal
from headroom.conformal.split import WINDOW, available_upto
from headroom.score.pinball import pinball_loss

#: Learning rate of the exponential weighting. Small enough that weights move over tens of
#: origins rather than jumping on a single one, large enough to have moved appreciably
#: within a shift lasting six weeks. `docs/methods.md` reports what it costs.
LEARNING_RATE: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class AggregatedConformal:
    """Adaptive conformal aggregated over a grid of step sizes.

    Attributes:
        alpha: The target miscoverage every expert aims at.
        gammas: The candidate step sizes.
        window: Calibration window in origins.
        learning_rate: Learning rate of the exponential weighting.
    """

    alpha: float
    gammas: tuple[float, ...] = GAMMA_GRID
    window: int = WINDOW
    learning_rate: float = LEARNING_RATE

    @property
    def name(self) -> str:
        """A label for the tables."""
        return f"aggregated conformal (alpha={self.alpha:.2f}, {len(self.gammas)} experts)"

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
        """Return the aggregated half-widths and the weight each expert carried.

        Args:
            scores: Nonconformity scores in origin order.
            horizon_step: Days ahead, counting from 1.
            origin_step: Days between origins.

        Returns:
            ``(widths, weights)``, shapes ``(n_origins,)`` and
            ``(n_origins, n_experts)``.
        """
        n = scores.size
        experts = [
            AdaptiveConformal(self.alpha, gamma, self.window).widths(
                scores, horizon_step, origin_step
            )
            for gamma in self.gammas
        ]
        panel = np.vstack(experts)  # (n_experts, n_origins)

        widths = np.full(n, np.nan)
        weights = np.full((n, len(self.gammas)), np.nan)

        cumulative = np.zeros(len(self.gammas))
        consumed = 0
        upper_level = 1.0 - self.alpha / 2.0

        for origin in range(n):
            end = available_upto(origin, horizon_step, origin_step)

            # Charge every expert for the outcomes that have become visible since the last
            # origin. The same ordering rule as the ACI update, for the same reason.
            while consumed < end:
                realised = scores[consumed]
                proposed = panel[:, consumed]
                if np.all(np.isfinite(proposed)):
                    cumulative += pinball_loss(proposed, realised, upper_level)
                consumed += 1

            if end == 0 or not np.all(np.isfinite(panel[:, origin])):
                continue

            weight = _softmin(cumulative, self.learning_rate)
            weights[origin] = weight
            widths[origin] = float(np.dot(weight, panel[:, origin]))

        return widths, weights


def _softmin(losses: npt.NDArray[np.float64], learning_rate: float) -> npt.NDArray[np.float64]:
    """Turn cumulative losses into weights, smallest loss weighted most.

    The maximum is subtracted before exponentiating, which changes nothing about the
    result and keeps it finite once cumulative losses have grown over a thousand origins.

    Args:
        losses: Cumulative loss per expert.
        learning_rate: Learning rate of the exponential weighting.

    Returns:
        Weights summing to 1, same shape as ``losses``.
    """
    scaled = -learning_rate * losses
    scaled -= scaled.max()
    weight = np.exp(scaled)
    total = weight.sum()
    if total <= 0.0 or not np.isfinite(total):  # pragma: no cover - guarded by the shift
        return np.full(losses.size, 1.0 / losses.size)
    return np.asarray(weight / total, dtype=np.float64)
