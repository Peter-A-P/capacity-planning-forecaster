"""TimesFM 2.5, a pretrained foundation model, used zero-shot on CPU.

PLAN.md section 2.7a. Every other model here is trained on these 37 series. This one is
not trained on them at all: the weights are Google's, downloaded once, and each origin is
inference. The question it answers is the one practitioners are asking now, which is
whether a pretrained model beats fitted baselines on demand it has never seen.

## The leak, and why this module knows about dates

TimesFM's weights were trained years after most of this backtest's origins, so a forecast
of March 2020 comes from a model that has already seen how series behaved in March 2020.
Every other model in this project honours the guarantee in :mod:`headroom.backtest.run`
that a forecast sees nothing after its origin; this one cannot. It is handled by scoring
it twice and never pooling the two: the full backtest, labelled as exposed, and the
**clean window** of origins whose whole horizon falls after the pretraining data ends.
:func:`first_clean_origin` is what finds that window, and :data:`PRETRAINING_ENDS` and
:data:`RELEASED` are the two dates it is drawn at. `docs/methods.md` records how the
corpus was checked and what could not be checked.

## What it sees

The same 1,095-day trailing window of demand as every other model, and nothing else: no
dates, no calendar, no holidays, no series identity. Each of the 37 series is forecast on
its own, so no information crosses between them at inference either.

## A median, and its own deciles beside it

TimesFM 2.5's quantile head stops at the 0.1 and 0.9 quantiles. This project scores on a
grid from 0.005 to 0.995 and reports 95 percent intervals, so taking its distribution as
the result would mean inventing the tails and making its CRPS depend on a rule chosen
here. Instead its **median** is the forecast, and its distribution is built by conformal
prediction from its own past errors, exactly as LightGBM's is. Its nine deciles are stored
beside the median because they cost nothing to keep, and they are what the one interval it
can produce unaided, the 80 percent, is reported from.

## Fixed settings, not tuned

Everything in :data:`FORECAST_CONFIG` is the model card's own configuration except the
context and horizon, which are this project's. Nothing was tuned on this backtest.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

import numpy as np
import numpy.typing as npt

#: The checkpoint, pinned in PLAN.md section 2.7a. Apache 2.0, unlike TimesFM 3.0.
MODEL_ID: Final[str] = "google/timesfm-2.5-200m-pytorch"

#: The name this model is stored and reported under.
NAME: Final[str] = "TimesFM"

#: The quantiles TimesFM 2.5 produces, and the grid its checkpoints are stored on.
DECILES: Final[npt.NDArray[np.float64]] = np.arange(1, 10) / 10.0

#: Where the median sits in :data:`DECILES`.
MEDIAN_AT: Final[int] = 4

#: Last day of the latest documented pretraining data, Wikimedia pageviews through
#: November 2023 (`docs/methods.md`). A forecast day after it is one the weights cannot
#: have been trained on.
PRETRAINING_ENDS: Final[date] = date(2023, 11, 30)

#: The day the weights were published. The most conservative reading of the same question,
#: kept as a cross-check on the primary window rather than as the result.
RELEASED: Final[date] = date(2025, 9, 15)

#: The forecast configuration, from the model card except for the two lines that are this
#: project's own: the 1,095-day window every model here is given, and the 14-day horizon.
FORECAST_CONFIG: Final[dict[str, Any]] = {
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "infer_is_positive": True,
    "fix_quantile_crossing": True,
}


def clean_from(cutoff: date) -> date:
    """The earliest origin day whose whole horizon falls after ``cutoff``.

    A forecast made at origin day ``d`` covers ``d + 1`` to ``d + horizon``, so its
    earliest forecast day is past the cutoff as soon as ``d`` reaches it. Every later day
    of that horizon is later still, which is why the horizon does not enter.

    Args:
        cutoff: The last day the weights may have been trained on.

    Returns:
        The first origin day of the clean window.
    """
    return cutoff


def clean_origins(origin_days: list[date], cutoff: date) -> int:
    """The number of the first origin in the clean window.

    Args:
        origin_days: Every origin's day, in order.
        cutoff: The last day the weights may have been trained on.

    Returns:
        The index into ``origin_days`` of the first clean origin.

    Raises:
        ValueError: No origin is clean. PLAN.md section 2.7a says that means TimesFM
            could not be evaluated fairly, not that the exposed numbers become the result.
    """
    first = clean_from(cutoff)
    clean = [i for i, day in enumerate(origin_days) if day >= first]
    if not clean:
        last = origin_days[-1] if origin_days else None
        raise ValueError(
            f"no clean window: the last origin is {last}, before {first}. PLAN.md 2.7a: "
            "report that TimesFM could not be evaluated fairly, not the exposed numbers"
        )
    return clean[0]


@dataclass(slots=True)
class ZeroShotTimesFM:
    """TimesFM 2.5, loaded once and asked for one forecast per origin.

    The weights are downloaded on first use and cached by `huggingface_hub`. Loading and
    compiling cost about half a minute and are done once per process, not per origin, so
    the per-origin time this records is inference alone.

    Attributes:
        horizon: Days forecast ahead.
        context: Days of history each forecast reads.
        name: The name used in the tables.
    """

    horizon: int
    context: int
    name: str = NAME
    _model: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Download or read the cached weights and compile for this context and horizon."""
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.context,
                max_horizon=self.horizon,
                **FORECAST_CONFIG,
            )
        )
        self._model = model

    def forecast(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Forecast the deciles of each day ahead for every series, zero-shot.

        Args:
            train: Demand up to and including the origin, shape ``(n_series, n_train)``.
                Only the last :attr:`context` days are read.
            horizon: Days to forecast; must equal :attr:`horizon`.
            levels: The quantile grid; must equal :data:`DECILES`, which is what the model
                produces.

        Returns:
            Quantile forecasts, shape ``(n_series, horizon, 9)``, sorted along the level
            axis and floored at zero.

        Raises:
            ValueError: ``horizon`` or ``levels`` differ from what the model produces.
        """
        if horizon != self.horizon:
            raise ValueError(f"built for a {self.horizon}-day horizon, asked for {horizon}")
        if not np.array_equal(levels, DECILES):
            raise ValueError("TimesFM 2.5 produces the deciles and no other grid")
        if self._model is None:
            self.load()

        window = train[:, -self.context :]
        # The library takes one array per series and returns the mean in column 0, then
        # the deciles. The mean is dropped: PLAN.md section 2.7a takes the median.
        _, quantiles = self._model.forecast(
            horizon=horizon, inputs=list(np.asarray(window, dtype=np.float32))
        )
        deciles = np.asarray(quantiles, dtype=np.float64)[..., 1:]
        np.clip(deciles, 0.0, None, out=deciles)
        return np.sort(deciles, axis=-1)
