"""Seasonal naive: the baseline every other number is reported against.

PLAN.md section 2.2. A project that skips this is not credible to anyone who has forecast
for a living, and the reason is that seasonal naive is genuinely hard to beat on daily
demand with a strong weekly cycle. Reporting a CRPS without it beside them says nothing
about whether the model did any work.

## The point forecast

The forecast for a day is what happened on the most recent same weekday that the
forecaster had already seen. For a horizon inside one week that is the value seven days
before the target; for the second week, fourteen days before. Nothing is fitted.

## The intervals

Seasonal naive is usually shown as a point forecast, which cannot be scored here. It is
given a predictive distribution the same way it gets its point forecast: from what
actually happened. The residuals of the same rule, applied across the training history at
the same horizon, are collected and their empirical quantiles are added to the point
forecast.

Two properties make this the right baseline rather than a strawman:

* The residual distribution is **per horizon**. A fourteen-day-ahead seasonal naive is
  more uncertain than a one-day-ahead one, and using one pooled residual distribution
  would hand every later model an easy win at short horizons.
* It is **empirical**, so it inherits whatever skew and heavy tails the series has.
  Assuming normal residuals on count data with a long right tail would make the baseline
  worse than it needs to be, and a baseline that has been quietly weakened makes every
  skill number in the project a lie.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from headroom.score.levels import check_levels

#: Days in the seasonal cycle. Emergency demand has a strong weekly shape, which is the
#: cycle a naive forecaster would use.
SEASON: Final[int] = 7


def seasonal_lag(horizon_step: int, season: int = SEASON) -> int:
    """Return how far back the seasonal naive rule reaches for one horizon step.

    One week back for days 1 to 7, two weeks for days 8 to 14, and so on, so that the
    value used is always one the forecaster had already seen at the origin.

    Args:
        horizon_step: Days ahead, counting from 1.
        season: Days in the seasonal cycle.

    Returns:
        The lag in days.

    Raises:
        ValueError: The horizon step is not positive.
    """
    if horizon_step < 1:
        raise ValueError(f"horizon step counts from 1, got {horizon_step}")
    return season * -(-horizon_step // season)  # season * ceil(step / season)


@dataclass(frozen=True, slots=True)
class SeasonalNaive:
    """The seasonal naive forecaster, with empirical residual quantiles.

    Attributes:
        season: Days in the seasonal cycle.
    """

    season: int = SEASON

    def forecast(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Forecast quantiles for every series.

        Args:
            train: History up to and including the origin, shape
                ``(n_series, n_train)``. Only the training slice may be passed; this
                method has no way to tell whether it was given future data.
            horizon: Days to forecast.
            levels: The quantile grid.

        Returns:
            Quantile forecasts, shape ``(n_series, horizon, n_levels)``, non-decreasing
            along the level axis.

        Raises:
            ValueError: The history is too short for the rule to reach back, or the
                levels are unusable.
        """
        check_levels(levels)
        if train.ndim != 2:
            raise ValueError(f"train must be (n_series, n_train), got {train.shape}")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        longest_lag = seasonal_lag(horizon, self.season)
        if train.shape[1] < 2 * longest_lag:
            raise ValueError(
                f"{train.shape[1]} days of history is too short: a {horizon}-day horizon "
                f"reaches back {longest_lag} days and needs at least that much again to "
                "estimate residuals"
            )

        n_series = train.shape[0]
        out = np.empty((n_series, horizon, levels.size))
        for step in range(1, horizon + 1):
            lag = seasonal_lag(step, self.season)
            point = train[:, -lag + (step - 1) % self.season]
            residual_quantiles = self._residual_quantiles(train, lag, levels)
            out[:, step - 1, :] = point[:, np.newaxis] + residual_quantiles

        # Counts cannot be negative. Clipping is a floor on the forecast, not a repair of
        # a crossing, so it cannot reorder the levels.
        np.clip(out, 0.0, None, out=out)
        return out

    def _residual_quantiles(
        self,
        train: npt.NDArray[np.float64],
        lag: int,
        levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Empirical quantiles of the rule's own residuals at one lag.

        The residual of the seasonal naive rule at lag ``k`` is just the ``k``-step
        seasonal difference of the series, which is why no fitting is involved.

        Args:
            train: History, shape ``(n_series, n_train)``.
            lag: The seasonal lag in days.
            levels: The quantile grid.

        Returns:
            Residual quantiles, shape ``(n_series, n_levels)``.
        """
        residuals = train[:, lag:] - train[:, :-lag]
        return np.quantile(residuals, levels, axis=1).T
