"""Statistical forecasters, through Nixtla StatsForecast.

PLAN.md section 2.2: ETS, Theta, AutoARIMA and MSTL for the double seasonality, all
producing prediction quantiles, all reported as skill against seasonal naive.

## The model sees no calendar

The training values are handed to StatsForecast against **synthetic consecutive dates**
starting at a fixed epoch, not the real ones. Nothing is lost: every model here takes its
seasonality from ``season_length`` positionally, and the panel is contiguous daily data
with no gaps (`docs/data.md`), so a weekly cycle is seven positions wherever it starts.

What is gained is that the guarantee in :mod:`headroom.backtest.run` stays literally true
for these models too. A model given real dates could in principle condition on knowing
that a particular origin is in March 2020. Given synthetic dates it cannot, because the
information is not there.

## Quantiles out of symmetric intervals

StatsForecast returns symmetric prediction intervals: a ``level`` of 90 gives ``lo-90``
and ``hi-90``, which are the 0.05 and 0.95 quantiles. Each requested quantile is therefore
mapped to the interval that contains it as a bound, ``level = |1 - 2q| * 100``, taking the
low bound below the median and the high bound above it. The median comes from the point
forecast. This needs no symmetry in the requested grid, only that each level round-trips
to an integer, which is checked rather than assumed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from headroom.score.levels import check_levels

#: Days in the weekly cycle, the seasonality every model here is given.
SEASON: Final[int] = 7

#: The annual cycle, for the model that takes more than one. 365 rather than 365.25
#: because the period has to be a whole number of observations.
YEAR: Final[int] = 365

#: Synthetic epoch the training values are dated from. Any fixed date does, because the
#: models read seasonality positionally; see the module docstring.
EPOCH: Final[date] = date(2000, 1, 1)

#: How close a derived StatsForecast level must be to an integer to be used.
_LEVEL_ATOL: Final[float] = 1e-9


def required_levels(levels: npt.NDArray[np.float64]) -> list[int]:
    """Return the StatsForecast interval levels a quantile grid needs.

    Args:
        levels: The quantile grid, strictly inside ``(0, 1)``.

    Returns:
        Integer levels, ascending, with the median needing none.

    Raises:
        ValueError: A quantile maps to a level that is not an integer, which
            StatsForecast cannot be asked for.
    """
    check_levels(levels)
    wanted: set[int] = set()
    for quantile in levels:
        if abs(quantile - 0.5) < _LEVEL_ATOL:
            continue
        level = abs(1.0 - 2.0 * float(quantile)) * 100.0
        rounded = round(level)
        if abs(level - rounded) > _LEVEL_ATOL * 100.0:
            raise ValueError(
                f"quantile {quantile} needs StatsForecast level {level}, which is not an "
                "integer; StatsForecast takes integer levels only"
            )
        wanted.add(int(rounded))
    return sorted(wanted)


def column_for(quantile: float, model: str) -> str:
    """Return the StatsForecast output column holding a quantile.

    Args:
        quantile: The quantile level.
        model: The model's name as StatsForecast reports it.

    Returns:
        The column name, for example ``"AutoETS-lo-90"``.
    """
    if abs(quantile - 0.5) < _LEVEL_ATOL:
        return model
    level = round(abs(1.0 - 2.0 * quantile) * 100.0)
    side = "lo" if quantile < 0.5 else "hi"
    return f"{model}-{side}-{level}"


@dataclass(frozen=True, slots=True)
class StatisticalModel:
    """One StatsForecast model, wrapped to the backtest's forecaster protocol.

    Attributes:
        name: The name used in the results tables.
        build: Builds the StatsForecast model object. A factory rather than an instance,
            because a fitted model must never be carried from one origin to the next.
        sf_name: The name StatsForecast gives the model in its output columns.
        n_jobs: Processes StatsForecast may use across series.
    """

    name: str
    build: Callable[[], Any]
    sf_name: str
    n_jobs: int = -1

    def forecast(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Fit on the training history and forecast quantiles.

        Args:
            train: History up to and including the origin, shape
                ``(n_series, n_train)``.
            horizon: Days to forecast.
            levels: The quantile grid.

        Returns:
            Quantile forecasts, shape ``(n_series, horizon, n_levels)``, floored at zero
            and sorted along the level axis.

        Raises:
            ValueError: The training array or the horizon is unusable.
        """
        from statsforecast import StatsForecast

        if train.ndim != 2:
            raise ValueError(f"train must be (n_series, n_train), got {train.shape}")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        n_series = train.shape[0]
        frame = _long_frame(train)
        engine = StatsForecast(models=[self.build()], freq="D", n_jobs=self.n_jobs)
        # StatsForecast returns whichever frame type it was handed. The input is
        # pandas so the output is too, but the signature is a union and mypy cannot
        # see which branch applies.
        raw = engine.forecast(h=horizon, df=frame, level=required_levels(levels))
        assert isinstance(raw, pd.DataFrame)
        out = raw.sort_values(["unique_id", "ds"])

        result = np.empty((n_series, horizon, levels.size))
        for j, quantile in enumerate(levels):
            column = out[column_for(float(quantile), self.sf_name)].to_numpy()
            result[:, :, j] = column.reshape(n_series, horizon)

        # A count cannot be negative, and a fitted model on a low-volume leaf will
        # happily predict one. Clipping is a floor, applied before the sort so it cannot
        # reorder anything.
        np.clip(result, 0.0, None, out=result)
        return np.sort(result, axis=-1)


def _long_frame(train: npt.NDArray[np.float64]) -> pd.DataFrame:
    """Build the long frame StatsForecast reads, on synthetic dates.

    Args:
        train: History, shape ``(n_series, n_train)``.

    Returns:
        A frame of ``unique_id``, ``ds`` and ``y``, sorted by series then date.
    """
    n_series, n_train = train.shape
    stamps = pd.date_range(EPOCH, periods=n_train, freq="D")
    return pd.DataFrame(
        {
            # Zero padded so string ordering matches series ordering, which is what the
            # reshape above relies on.
            "unique_id": np.repeat([f"s{i:04d}" for i in range(n_series)], n_train),
            "ds": np.tile(stamps.to_numpy(), n_series),
            "y": train.reshape(-1),
        }
    )


def catalogue(season: int = SEASON, year: int = YEAR) -> list[StatisticalModel]:
    """Return the statistical models PLAN.md section 2.2 names, in reporting order.

    Args:
        season: The weekly cycle.
        year: The annual cycle, for MSTL.

    Returns:
        The models.
    """
    from statsforecast.models import MSTL, AutoARIMA, AutoETS, AutoTheta

    return [
        StatisticalModel("ETS", lambda: AutoETS(season_length=season), "AutoETS"),
        StatisticalModel("Theta", lambda: AutoTheta(season_length=season), "AutoTheta"),
        StatisticalModel("MSTL", lambda: MSTL(season_length=[season, year]), "MSTL"),
        StatisticalModel("AutoARIMA", lambda: AutoARIMA(season_length=season), "AutoARIMA"),
    ]


#: Models that are dropped first if the schedule tightens, most expendable first.
#: PLAN.md section 5 names the order; AutoARIMA is by far the most expensive per fit and
#: `docs/methods.md` reports what it costs and whether it earned it.
DROP_ORDER: Final[tuple[str, ...]] = ("AutoARIMA", "MSTL", "Theta")


@dataclass(frozen=True, slots=True)
class BatchedModels:
    """Several StatsForecast models fitted in one call per origin.

    Fitting three models in one call costs measurably less than three calls: the frame is
    built once and StatsForecast's process pool is started once. On this panel four models
    together took 190 seconds against 275 if their individual times are added, so batching
    saves about 31 percent, which is the difference between a ten-hour run and a fourteen
    hour one.

    Attributes:
        models: The models to fit together.
        n_jobs: Processes StatsForecast may use across series.
    """

    models: tuple[StatisticalModel, ...]
    n_jobs: int = -1

    @property
    def names(self) -> tuple[str, ...]:
        """The models' names, in order."""
        return tuple(model.name for model in self.models)

    def forecast_all(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        levels: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Fit every model on the same history and return each one's quantiles.

        Args:
            train: History up to and including the origin, shape
                ``(n_series, n_train)``.
            horizon: Days to forecast.
            levels: The quantile grid.

        Returns:
            One array per model name, each shape ``(n_series, horizon, n_levels)``.

        Raises:
            ValueError: The training array or the horizon is unusable.
        """
        from statsforecast import StatsForecast

        if train.ndim != 2:
            raise ValueError(f"train must be (n_series, n_train), got {train.shape}")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        n_series = train.shape[0]
        engine = StatsForecast(
            models=[model.build() for model in self.models], freq="D", n_jobs=self.n_jobs
        )
        raw = engine.forecast(h=horizon, df=_long_frame(train), level=required_levels(levels))
        assert isinstance(raw, pd.DataFrame)
        out = raw.sort_values(["unique_id", "ds"])

        results: dict[str, npt.NDArray[np.float64]] = {}
        for model in self.models:
            block = np.empty((n_series, horizon, levels.size))
            for j, quantile in enumerate(levels):
                column = out[column_for(float(quantile), model.sf_name)].to_numpy()
                block[:, :, j] = column.reshape(n_series, horizon)
            np.clip(block, 0.0, None, out=block)
            results[model.name] = np.sort(block, axis=-1)
        return results
