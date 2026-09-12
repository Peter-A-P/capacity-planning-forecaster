"""Probabilistic scores.

PLAN.md section 2.3: CRPS, pinball loss, coverage and width are the result. MAE is a
footnote for readers who look for it and is never the headline, because a point forecast
cannot be scored on the thing that matters here, which is whether the range held.

Every function takes quantile forecasts and observations as plain arrays and returns
scores as plain arrays. Nothing here knows about the hierarchy, the backtest or a model.
"""

from headroom.score.crps import crps_from_quantiles, crps_normal
from headroom.score.pinball import pinball_by_level, pinball_loss

__all__ = ["crps_from_quantiles", "crps_normal", "pinball_by_level", "pinball_loss"]
