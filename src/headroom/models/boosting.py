"""A global LightGBM model, to separate learning across series from deep learning.

PLAN.md section 2.7b. The statistical models are fitted one series at a time with no
features; the neural models are global and deep. A global gradient-boosted model is global
without being deep, so whatever it gains over ETS is the gain from learning across series,
and whatever the neural models gain over it is the gain from the architecture.

## What the model may see

Demand values up to and including the origin, as every model here, and **the dates** of
the training window and of the fourteen days being forecast. Dates are known in advance and
carry no demand, so this does not weaken the guarantee in :mod:`headroom.backtest.run`; it
is what lets the model use day of week and public holidays for the day it is forecasting.
It is the only model that sees holidays, and every table that reports it says so.

No feature identifies the year, so the model cannot condition on an origin being in 2020.

## One row per series, anchor day and horizon step

The horizon is **direct**: one model with the horizon step as a feature, trained on rows
``(series, anchor day t, step h)`` whose target is the value ``h`` days after ``t``. Every
lag in a row is read at or before ``t``, so a forecast never feeds on its own earlier
steps and errors do not compound. :func:`design` builds the rows and
`tests/test_boosting.py` corrupts every value after an anchor and asserts that no feature
of that anchor's rows changes.

## One scale per series

The city averages about 3,800 incidents a day and a dispatch area about 20. A global model
on raw counts would spend its whole capacity on the city. Each series is divided by its own
mean over the training window, which is known at the origin, and forecasts are multiplied
back.

## A median, not a distribution

The model is fitted with an absolute-error objective, so it forecasts a median. Its
predictive distribution is built afterwards from its own out-of-sample errors at earlier
origins, in :mod:`headroom.conformal.predictive`, rather than by fitting one quantile model
per level of a 199-level grid.

## Fixed settings, not tuned

The booster's settings are fixed below and were not tuned on this backtest. Tuning them on
the same origins the model is scored on would be a quiet leak of its own.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from headroom.data.calendar import features as calendar_features
from headroom.types import LEVELS, Level

#: Days in the weekly cycle.
SEASON: Final[int] = 7

#: Same-weekday values used per row. Four is the last month of that weekday.
N_SEASONAL_LAGS: Final[int] = 4

#: Recent daily values, as days before the anchor (0 is the anchor day itself).
RECENT_LAGS: Final[tuple[int, ...]] = (0, 1, 6, 13)

#: Trailing windows for rolling means, in days ending at the anchor.
ROLLING_WINDOWS: Final[tuple[int, ...]] = (7, 28, 91)

#: Earliest anchor with every feature defined: the longest look-back in any row.
LOOKBACK: Final[int] = max(
    max(ROLLING_WINDOWS),
    max(RECENT_LAGS) + 1,
    # Same-weekday lags reach furthest for steps 1 to 7: step 1 reads 27 days before its
    # anchor, which is 28 days counting the anchor.
    SEASON * N_SEASONAL_LAGS,
)

FEATURES: Final[tuple[str, ...]] = (
    "step",
    "series",
    "level",
    *(f"same_weekday_{k}" for k in range(1, N_SEASONAL_LAGS + 1)),
    *(f"recent_{lag}" for lag in RECENT_LAGS),
    *(f"mean_{window}" for window in ROLLING_WINDOWS),
    "dow",
    "day_of_year",
    "is_holiday",
    "is_holiday_observed",
)

#: Features LightGBM treats as categories rather than numbers.
CATEGORICAL: Final[tuple[str, ...]] = ("series", "level", "dow")

#: The booster's settings. Fixed, and not tuned on the backtest (module docstring).
#: ``num_threads`` is the machine's physical core count. It changes speed only: six threads
#: and the default gave identical forecasts. Which is faster on an idle machine has not been
#: measured; the comparison once recorded here was taken on a busy one (`docs/methods.md`).
PARAMS: Final[dict[str, Any]] = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 6,
    "verbose": -1,
}

#: Boosting rounds. Fixed with :data:`PARAMS`.
ROUNDS: Final[int] = 400


def _calendar(days: tuple[date, ...]) -> npt.NDArray[np.float64]:
    """Return the calendar features for a run of days.

    Args:
        days: The days, ascending.

    Returns:
        Shape ``(len(days), 4)``: day of week, day of year, holiday, observed holiday.
    """
    frame = calendar_features(days)
    return np.column_stack(
        [
            frame["dow"].to_numpy().astype(np.float64),
            np.array([d.timetuple().tm_yday for d in days], dtype=np.float64),
            frame["is_holiday"].to_numpy().astype(np.float64),
            frame["is_holiday_observed"].to_numpy().astype(np.float64),
        ]
    )


def design(
    scaled: npt.NDArray[np.float64],
    calendar: npt.NDArray[np.float64],
    anchors: npt.NDArray[np.intp],
    step: int,
    level_codes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the feature rows for one horizon step at a set of anchor days.

    Every lag is read at or before its anchor. The calendar is read at the target day,
    ``anchor + step``, which is a date and never a demand value.

    Args:
        scaled: Demand divided by each series' scale, shape ``(n_series, n_days)``.
        calendar: Calendar features for every day in ``scaled`` and for the days after it
            that any target can fall on, shape ``(n_days_with_horizon, 4)``.
        anchors: Anchor day indices into ``scaled``, each at least :data:`LOOKBACK` - 1.
        step: Days from the anchor to the target, counting from 1.
        level_codes: The hierarchy level of each series as a number, shape ``(n_series,)``.

    Returns:
        Feature rows in :data:`FEATURES` order, shape ``(n_series * len(anchors),
        len(FEATURES))``, series outermost.

    Raises:
        ValueError: An anchor is too early for its look-back.
    """
    if anchors.size and anchors.min() < LOOKBACK - 1:
        raise ValueError(f"anchors must be at least {LOOKBACK - 1}, got {anchors.min()}")
    n_series = scaled.shape[0]
    n_anchors = anchors.size
    columns: list[npt.NDArray[np.float64]] = []

    def per_series(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return values.reshape(n_series * n_anchors)

    def broadcast(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.broadcast_to(values, (n_series, n_anchors)).reshape(-1)

    columns.append(np.full(n_series * n_anchors, float(step)))
    columns.append(np.repeat(np.arange(n_series, dtype=np.float64), n_anchors))
    columns.append(np.repeat(level_codes, n_anchors))

    first_week = -(-step // SEASON)  # the most recent same weekday already observed
    for k in range(first_week, first_week + N_SEASONAL_LAGS):
        columns.append(per_series(scaled[:, anchors + step - SEASON * k]))
    for lag in RECENT_LAGS:
        columns.append(per_series(scaled[:, anchors - lag]))

    cumulative = np.concatenate([np.zeros((n_series, 1)), np.cumsum(scaled, axis=1)], axis=1)
    for window in ROLLING_WINDOWS:
        total = cumulative[:, anchors + 1] - cumulative[:, anchors + 1 - window]
        columns.append(per_series(total / window))

    target_calendar = calendar[anchors + step]
    for j in range(target_calendar.shape[1]):
        columns.append(broadcast(target_calendar[:, j]))

    return np.column_stack(columns)


@dataclass(frozen=True, slots=True)
class GlobalLightGBM:
    """One LightGBM model across every series, refitted at every origin.

    Attributes:
        levels: The hierarchy level of each series, in panel row order.
        name: The name used in the results tables.
        params: LightGBM settings; see :data:`PARAMS`.
        rounds: Boosting rounds; see :data:`ROUNDS`.
    """

    levels: tuple[Level, ...]
    name: str = "LightGBM"
    params: dict[str, Any] = field(default_factory=lambda: dict(PARAMS))
    rounds: int = ROUNDS

    def forecast(
        self,
        train: npt.NDArray[np.float64],
        horizon: int,
        days: tuple[date, ...],
    ) -> npt.NDArray[np.float64]:
        """Fit on the training window and forecast the median of each day ahead.

        Args:
            train: Demand up to and including the origin, shape ``(n_series, n_train)``.
            horizon: Days to forecast.
            days: The dates of the training window followed by the ``horizon`` days
                being forecast, so ``len(days) == n_train + horizon``.

        Returns:
            Median forecasts, shape ``(n_series, horizon)``, floored at zero.

        Raises:
            ValueError: The inputs do not line up or the window is too short.
        """
        import lightgbm as lgb

        n_series, n_train = train.shape
        if len(days) != n_train + horizon:
            raise ValueError(
                f"{len(days)} dates for {n_train} training days and a {horizon}-day horizon"
            )
        if len(self.levels) != n_series:
            raise ValueError(f"{len(self.levels)} levels for {n_series} series")
        if n_train < LOOKBACK + horizon:
            raise ValueError(f"{n_train} training days is shorter than {LOOKBACK + horizon}")

        scale = np.maximum(train.mean(axis=1), 1.0)
        scaled = train / scale[:, np.newaxis]
        calendar = _calendar(days)
        level_codes = np.array([LEVELS.index(level) for level in self.levels], dtype=float)

        rows: list[npt.NDArray[np.float64]] = []
        targets: list[npt.NDArray[np.float64]] = []
        for step in range(1, horizon + 1):
            anchors = np.arange(LOOKBACK - 1, n_train - step, dtype=np.intp)
            rows.append(design(scaled, calendar, anchors, step, level_codes))
            targets.append(scaled[:, anchors + step].reshape(-1))

        data = lgb.Dataset(
            np.vstack(rows),
            label=np.concatenate(targets),
            feature_name=list(FEATURES),
            categorical_feature=list(CATEGORICAL),
            free_raw_data=True,
        )
        booster = lgb.train(self.params, data, num_boost_round=self.rounds)

        origin = np.array([n_train - 1], dtype=np.intp)
        future = np.vstack(
            [
                design(scaled, calendar, origin, step, level_codes)
                for step in range(1, horizon + 1)
            ]
        )
        predicted = np.asarray(booster.predict(future), dtype=np.float64)
        # Rows are step outermost, then series; reorder to (series, step).
        median = predicted.reshape(horizon, n_series).T * scale[:, np.newaxis]
        return np.clip(median, 0.0, None)
