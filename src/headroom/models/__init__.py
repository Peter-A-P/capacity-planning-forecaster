"""Forecasters, all producing quantiles.

Every model here takes a training array and returns quantile forecasts on the same grid,
so the backtest and the scores never know which model made them. PLAN.md section 2.2:
seasonal naive comes first and everything else is reported as skill against it.
"""

from headroom.models.baselines import SeasonalNaive

__all__ = ["SeasonalNaive"]
