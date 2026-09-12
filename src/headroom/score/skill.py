"""Skill: every score reported relative to seasonal naive.

PLAN.md section 2.2 and the repository's own rules make this the reporting unit. A CRPS
of 240 means nothing to a reader; a CRPS 12 percent below the score of the forecast any
practitioner would try first means something, and it is the only form in which a number
from this project can be compared with a number from anyone else's.

Skill is ``1 - score / baseline_score``, so it is positive when the method beats the
baseline, zero when it ties, and unbounded below. It is defined for scores where lower is
better, which is all of them here.
"""

import numpy as np
import numpy.typing as npt


def skill(
    score: npt.NDArray[np.float64] | float,
    baseline: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """Skill of a score against a baseline score.

    Args:
        score: The method's score. Lower is better.
        baseline: The baseline's score on the same data. Lower is better, strictly
            positive.

    Returns:
        ``1 - score / baseline``. Positive means the method beat the baseline.

    Raises:
        ValueError: A baseline score is zero or negative, which leaves skill undefined.
    """
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    if np.any(baseline_arr <= 0.0):
        raise ValueError(
            "skill is undefined against a baseline score of zero or less; a baseline "
            "that scored zero was perfect and nothing can improve on it proportionally"
        )
    result: npt.NDArray[np.float64] = 1.0 - np.asarray(score, dtype=np.float64) / baseline_arr
    return result


def paired_difference(
    score: npt.NDArray[np.float64], other: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Per-origin score difference between two methods, ``other - score``.

    Paired, because the two methods saw the same origins and the same data. The variation
    between origins is enormous next to the variation between methods, and differencing
    removes it; comparing two unpaired means would put that variation in the interval and
    hide a real difference. PLAN.md section 1 requires the neural comparison to be paired
    for exactly this reason.

    Args:
        score: One method's per-origin scores.
        other: The other method's per-origin scores, same shape and same origins.

    Returns:
        ``other - score``, so a positive value means ``score``'s method was better.

    Raises:
        ValueError: The two are not the same shape, which would mean they are not paired.
    """
    if score.shape != other.shape:
        raise ValueError(
            f"paired scores must have the same shape, got {score.shape} and {other.shape}"
        )
    return other - score
