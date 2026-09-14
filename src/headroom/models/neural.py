"""N-HiTS, a global neural forecaster, through Nixtla NeuralForecast on CPU.

PLAN.md section 2.7: a named neural model, trained across all 37 series, with a
multi-quantile loss, allowed to lose. It is compared paired against ETS, the best
statistical model, and against the global LightGBM model, which already showed that
learning across series with a holiday calendar buys nothing over ETS on this panel
(`docs/methods.md`). Whatever N-HiTS gains over LightGBM is what the architecture buys.

## What it sees

Demand only, on **synthetic consecutive dates**, exactly as the statistical models in
:mod:`headroom.models.stats` do. It has no calendar and no holidays, so it has less
information than LightGBM, and every table that reports it can say so.

## Quantiles

The loss is the multi-quantile loss on the project's own 199-level scoring grid, so the
network's output is the distribution CRPS is scored on and no conformal step is needed to
score it. Quantile heads can cross; they are sorted and floored at zero, as for every other
model, and the crossing rate is reported by the scorer.

## Fitting is separate from forecasting

A fit costs far more than a forecast, so the model is fitted with :meth:`NHiTS.fit` and
then forecasts from any later window with :meth:`NHiTS.predict` without refitting. The
backtest decides how often to refit (:mod:`headroom.backtest.origins`); a forecast made
between refits uses weights trained only on data before the last refit's origin, so it
is never trained on anything after its own origin either.

## Fixed settings, not tuned

Every setting below is fixed and was not tuned on this backtest, for the reason given in
:mod:`headroom.models.boosting`.
"""

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from headroom.models.stats import _long_frame
from headroom.score.levels import check_levels

#: Days of history each forecast reads: sixteen weeks, eight times the horizon.
INPUT_SIZE: Final[int] = 112

#: The network and its training. NeuralForecast's N-HiTS defaults except where stated.
SETTINGS: Final[dict[str, Any]] = {
    "input_size": INPUT_SIZE,
    "max_steps": 1000,
    "learning_rate": 1e-3,
    "batch_size": 37,  # every series in every batch
    "windows_batch_size": 1024,
    "scaler_type": "robust",  # per-window scaling, so the city does not swamp the areas
    "random_seed": 0,
    "accelerator": "cpu",
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "enable_checkpointing": False,
}


@dataclass(slots=True)
class NHiTS:
    """N-HiTS with a multi-quantile loss, fitted once and forecasting many times.

    Attributes:
        levels: The quantile grid the loss is trained on and forecasts are returned on.
        horizon: Days forecast ahead.
        name: The name used in the results tables.
        settings: NeuralForecast settings; see :data:`SETTINGS`.
    """

    levels: npt.NDArray[np.float64]
    horizon: int
    name: str = "N-HiTS"
    settings: dict[str, Any] = field(default_factory=lambda: dict(SETTINGS))
    _engine: Any = field(default=None, init=False, repr=False)

    def fit(self, train: npt.NDArray[np.float64]) -> None:
        """Train on a window of history, replacing any earlier fit.

        Args:
            train: Demand up to and including the refit origin, shape
                ``(n_series, n_train)``.

        Raises:
            ValueError: The window is too short to hold one training example.
        """
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS

        check_levels(self.levels)
        if train.ndim != 2 or train.shape[1] < self.settings["input_size"] + self.horizon:
            raise ValueError(
                f"train {train.shape} is too short for input {self.settings['input_size']} "
                f"and horizon {self.horizon}"
            )
        model = NHITS(
            h=self.horizon,
            loss=MQLoss(quantiles=[float(q) for q in self.levels]),
            alias=self.name,
            **self.settings,
        )
        engine = NeuralForecast(models=[model], freq="D")
        engine.fit(df=_long_frame(train), val_size=0)
        self._engine = engine

    def predict(self, train: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forecast from the end of a window with the weights of the last fit.

        Args:
            train: Demand up to and including the forecast origin, shape
                ``(n_series, n_days)``. Only the last ``input_size`` days are read.

        Returns:
            Quantile forecasts, shape ``(n_series, horizon, n_levels)``, sorted along the
            level axis and floored at zero.

        Raises:
            RuntimeError: :meth:`fit` has not been called.
        """
        from neuralforecast.losses.pytorch import quantiles_to_outputs

        if self._engine is None:
            raise RuntimeError("fit before predicting")
        n_series = train.shape[0]
        raw = self._engine.predict(df=_long_frame(train))
        assert isinstance(raw, pd.DataFrame)
        out = raw.reset_index().sort_values(["unique_id", "ds"])

        _, suffixes = quantiles_to_outputs([float(q) for q in self.levels])
        result = np.empty((n_series, self.horizon, self.levels.size))
        for j, suffix in enumerate(suffixes):
            column = out[f"{self.name}{suffix}"].to_numpy(dtype=np.float64)
            result[:, :, j] = column.reshape(n_series, self.horizon)
        np.clip(result, 0.0, None, out=result)
        return np.sort(result, axis=-1)

    def forecast(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Fit and forecast in one call, for the backtest's forecaster protocol.

        Args:
            train: Demand up to and including the origin, shape ``(n_series, n_train)``.
            horizon: Days to forecast; must equal :attr:`horizon`.
            levels: The quantile grid; must equal :attr:`levels`.

        Returns:
            Quantile forecasts, shape ``(n_series, horizon, n_levels)``.

        Raises:
            ValueError: ``horizon`` or ``levels`` differ from the model's own.
        """
        if horizon != self.horizon or not np.array_equal(levels, self.levels):
            raise ValueError("an N-HiTS model is built for one horizon and one quantile grid")
        self.fit(train)
        return self.predict(train)
