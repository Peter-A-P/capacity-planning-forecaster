"""Pinball loss, the score a single quantile is judged by.

The pinball loss at level ``tau`` is minimised, in expectation, by the true ``tau``
quantile. That is the whole reason quantile forecasts can be scored at all, and it is
also the reason CRPS can be written as an integral of this over ``tau``
(:mod:`headroom.score.crps`).
"""

import numpy as np
import numpy.typing as npt

from headroom.score.levels import check_levels


def pinball_loss(
    forecast: npt.NDArray[np.float64] | float,
    observed: npt.NDArray[np.float64] | float,
    level: float,
) -> npt.NDArray[np.float64]:
    """Pinball loss of one quantile forecast.

    Under-forecasting is charged ``level`` per unit and over-forecasting ``1 - level``,
    which is what tilts the minimiser to the ``level`` quantile: at 0.95, being 1 under
    costs nineteen times being 1 over.

    Args:
        forecast: The predicted quantile, any shape.
        observed: The realised value, broadcastable to ``forecast``.
        level: The quantile level, strictly between 0 and 1.

    Returns:
        The loss, elementwise, same shape as the broadcast.

    Raises:
        ValueError: The level is not strictly between 0 and 1.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    error = np.asarray(observed, dtype=np.float64) - np.asarray(forecast, dtype=np.float64)
    loss: npt.NDArray[np.float64] = np.maximum(level * error, (level - 1.0) * error)
    return loss


def pinball_by_level(
    quantiles: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pinball loss at every level of a quantile forecast.

    Args:
        quantiles: Predicted quantiles, shape ``(..., n_levels)``, with the level axis
            last and increasing along it.
        observed: Realised values, shape ``(...)``, broadcast across the level axis.
        levels: The quantile levels, strictly increasing inside ``(0, 1)``.

    Returns:
        The loss, shape ``(..., n_levels)``.

    Raises:
        ValueError: The levels are unusable, or the last axis does not match them.
    """
    check_levels(levels)
    if quantiles.shape[-1] != levels.size:
        raise ValueError(
            f"quantiles have {quantiles.shape[-1]} levels on the last axis, "
            f"levels has {levels.size}"
        )
    error = observed[..., np.newaxis] - quantiles
    return np.maximum(levels * error, (levels - 1.0) * error)
