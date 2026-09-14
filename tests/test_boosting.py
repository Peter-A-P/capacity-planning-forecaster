"""The global LightGBM model and the rows it learns from.

The test that matters most here is the first one. PLAN.md section 8 names the risk: a lag
or rolling feature that reaches one day past its anchor flatters the model silently and
says nothing about it in the output. So every value after an anchor is corrupted and no
feature of that anchor's rows may move.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from headroom.models.boosting import (
    FEATURES,
    LOOKBACK,
    PARAMS,
    GlobalLightGBM,
    _calendar,
    design,
)

HORIZON = 14
N_SERIES = 3
N_DAYS = 240
LEVELS = np.array([0.0, 1.0, 2.0])
COLUMN = {name: j for j, name in enumerate(FEATURES)}


def _panel(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.5, 1.5, size=(N_SERIES, N_DAYS))


def _calendar_rows(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(N_DAYS + HORIZON, 4))


@pytest.mark.parametrize("step", range(1, HORIZON + 1))
def test_no_feature_reads_a_value_after_its_anchor(step):
    scaled = _panel()
    calendar = _calendar_rows()
    anchor = np.array([150], dtype=np.intp)
    before = design(scaled, calendar, anchor, step, LEVELS)

    corrupted = scaled.copy()
    corrupted[:, anchor[0] + 1 :] = 1e9
    after = design(corrupted, calendar, anchor, step, LEVELS)

    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("step", [1, 7, 8, 14])
def test_the_calendar_is_read_at_the_target_day_and_nowhere_else(step):
    scaled = _panel()
    calendar = np.zeros((N_DAYS + HORIZON, 4))
    anchor = 150
    calendar[anchor + step, 2] = 1.0  # a holiday on the target day only
    rows = design(scaled, calendar, np.array([anchor], dtype=np.intp), step, LEVELS)
    assert np.all(rows[:, COLUMN["is_holiday"]] == 1.0)

    calendar[:, :] = 0.0
    calendar[anchor, 2] = 1.0  # a holiday on the anchor day is not the target's
    rows = design(scaled, calendar, np.array([anchor], dtype=np.intp), step, LEVELS)
    assert np.all(rows[:, COLUMN["is_holiday"]] == 0.0)


def test_same_weekday_lags_are_the_values_seasonal_naive_would_use():
    scaled = np.tile(np.arange(N_DAYS, dtype=np.float64), (N_SERIES, 1))
    calendar = _calendar_rows()
    anchor = 150
    for step, nearest in ((1, 7), (7, 7), (8, 14), (14, 14)):
        rows = design(scaled, calendar, np.array([anchor], dtype=np.intp), step, LEVELS)
        assert rows[0, COLUMN["same_weekday_1"]] == anchor + step - nearest
        assert rows[0, COLUMN["same_weekday_4"]] == anchor + step - nearest - 21


def test_rolling_means_end_at_the_anchor():
    scaled = _panel()
    anchor = 150
    rows = design(scaled, _calendar_rows(), np.array([anchor], dtype=np.intp), 3, LEVELS)
    for window in (7, 28, 91):
        expected = scaled[:, anchor + 1 - window : anchor + 1].mean(axis=1)
        np.testing.assert_allclose(rows[:, COLUMN[f"mean_{window}"]], expected)


def test_rows_are_series_outermost_with_their_identity_and_level():
    anchors = np.array([100, 101], dtype=np.intp)
    rows = design(_panel(), _calendar_rows(), anchors, 5, LEVELS)
    assert rows.shape == (N_SERIES * anchors.size, len(FEATURES))
    np.testing.assert_array_equal(rows[:, COLUMN["series"]], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(rows[:, COLUMN["level"]], [0, 0, 1, 1, 2, 2])
    assert np.all(rows[:, COLUMN["step"]] == 5)


def test_an_anchor_without_the_full_lookback_is_refused():
    with pytest.raises(ValueError, match="anchors must be at least"):
        design(_panel(), _calendar_rows(), np.array([LOOKBACK - 2], dtype=np.intp), 1, LEVELS)


def test_calendar_features_mark_a_real_holiday():
    days = tuple(date(2024, 7, 1) + timedelta(days=i) for i in range(7))
    calendar = _calendar(days)
    independence = days.index(date(2024, 7, 4))
    assert calendar[independence, 2] == 1.0
    assert calendar[independence, 0] == 3.0  # a Thursday, Monday being 0
    assert calendar[independence, 1] == 186.0  # 2024 is a leap year


# --- the fitted model -----------------------------------------------------------------


def _weekly_panel(n_days: int = 500, seed: int = 0) -> tuple[np.ndarray, tuple[date, ...]]:
    rng = np.random.default_rng(seed)
    shape = np.array([1.3, 1.1, 1.0, 0.95, 0.9, 0.8, 0.95])
    days = tuple(date(2015, 1, 5) + timedelta(days=i) for i in range(n_days + HORIZON))
    scale = np.array([4000.0, 800.0, 20.0])[:, np.newaxis]
    weekly = shape[np.arange(n_days + HORIZON) % 7]
    values = scale * weekly * rng.normal(1.0, 0.03, size=(3, n_days + HORIZON))
    return values, days


def _small_model() -> GlobalLightGBM:
    return GlobalLightGBM(
        levels=("city", "borough", "area"), params={**PARAMS, "num_threads": 2}, rounds=60
    )


def test_the_model_learns_a_weekly_shape_shared_across_scales():
    values, days = _weekly_panel()
    n_train = values.shape[1] - HORIZON
    median = _small_model().forecast(values[:, :n_train], HORIZON, days)
    truth = values[:, n_train:]
    flat = values[:, n_train - 28 : n_train].mean(axis=1, keepdims=True)

    assert median.shape == (3, HORIZON)
    assert np.all(median >= 0.0)
    relative_error = np.abs(median - truth) / truth
    flat_error = np.abs(flat - truth) / truth
    assert relative_error.mean() < 0.5 * flat_error.mean()


def test_the_same_inputs_give_the_same_forecast():
    values, days = _weekly_panel(n_days=300)
    n_train = values.shape[1] - HORIZON
    first = _small_model().forecast(values[:, :n_train], HORIZON, days)
    second = _small_model().forecast(values[:, :n_train], HORIZON, days)
    np.testing.assert_array_equal(first, second)


def test_dates_that_do_not_cover_the_horizon_are_refused():
    values, days = _weekly_panel(n_days=300)
    n_train = values.shape[1] - HORIZON
    with pytest.raises(ValueError, match="dates for"):
        _small_model().forecast(values[:, :n_train], HORIZON, days[:-1])
