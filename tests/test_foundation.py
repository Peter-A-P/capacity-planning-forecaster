"""TimesFM zero-shot: the clean window, the grid it is stored on, and the forecast itself.

The forecasting test needs the weights, so it is marked slow and skipped by default; the
rest are about the leak, which is where this model can go wrong quietly. PLAN.md section
2.7a is the specification these enforce.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from headroom.backtest.origins import Origins
from headroom.backtest.store import open_checkpoint
from headroom.cli import STORED_LEVELS, _median_at, _stored_levels
from headroom.models.foundation import (
    DECILES,
    MEDIAN_AT,
    NAME,
    PRETRAINING_ENDS,
    RELEASED,
    ZeroShotTimesFM,
    clean_from,
    clean_origins,
)

HORIZON = 14
WEEKLY = [date(2023, 1, 4) + timedelta(days=7 * i) for i in range(200)]


def test_the_clean_window_starts_at_the_earliest_origin_whose_horizon_is_clean():
    # An origin on the cutoff day forecasts the day after it onward, which is clean; the
    # day before it forecasts the cutoff day itself, which is not.
    first = clean_from(PRETRAINING_ENDS)
    assert first + timedelta(days=1) > PRETRAINING_ENDS
    assert (first - timedelta(days=1)) + timedelta(days=1) <= PRETRAINING_ENDS


def test_every_day_of_a_clean_origins_horizon_is_after_the_cutoff():
    at = clean_origins(WEEKLY, PRETRAINING_ENDS)
    origin = WEEKLY[at]
    horizon = [origin + timedelta(days=h) for h in range(1, HORIZON + 1)]
    assert min(horizon) > PRETRAINING_ENDS


def test_the_origin_before_the_clean_window_is_not_clean():
    at = clean_origins(WEEKLY, PRETRAINING_ENDS)
    assert at > 0, "the fixture should straddle the cutoff"
    assert WEEKLY[at - 1] + timedelta(days=1) <= PRETRAINING_ENDS


def test_the_release_cross_check_is_a_later_start_on_the_same_origins():
    # The cross-check is the more conservative reading of the same question, so it can only
    # ever start later, which is what makes it a subset and not a second result.
    assert RELEASED > PRETRAINING_ENDS
    assert clean_origins(WEEKLY, RELEASED) > clean_origins(WEEKLY, PRETRAINING_ENDS)


def test_a_panel_that_ends_before_the_cutoff_has_no_clean_window():
    # PLAN.md 2.7a: that is reported as "could not be evaluated fairly", never as the
    # exposed numbers, so this has to fail loudly rather than return an empty window.
    stale = [date(2019, 1, 1) + timedelta(days=7 * i) for i in range(10)]
    with pytest.raises(ValueError, match="no clean window"):
        clean_origins(stale, PRETRAINING_ENDS)


def test_the_stored_grid_is_the_deciles_the_model_actually_produces():
    assert _stored_levels(NAME) is DECILES
    assert np.allclose(DECILES, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_the_median_column_is_where_the_scorer_looks_for_it():
    # _distribution reads the median out of the stored grid; if MEDIAN_AT and the scorer's
    # own arithmetic disagreed, TimesFM would be scored on its 0.4 or 0.6 quantile and
    # nothing else would notice.
    assert _median_at(_stored_levels(NAME)) == MEDIAN_AT
    assert DECILES[MEDIAN_AT] == 0.5


def test_timesfm_is_scored_conformally_because_its_grid_is_not_the_scoring_grid():
    assert NAME in STORED_LEVELS
    assert STORED_LEVELS[NAME].size < 199


def test_a_grid_the_model_cannot_produce_is_refused():
    model = ZeroShotTimesFM(horizon=HORIZON, context=1_095)
    with pytest.raises(ValueError, match="deciles"):
        model.forecast(np.ones((3, 1_095)), HORIZON, np.array([0.05, 0.5, 0.95]))


def test_a_horizon_it_was_not_compiled_for_is_refused():
    model = ZeroShotTimesFM(horizon=HORIZON, context=1_095)
    with pytest.raises(ValueError, match="14-day horizon"):
        model.forecast(np.ones((3, 1_095)), 7, DECILES)


def test_a_decile_checkpoint_saves_and_resumes_on_its_own_grid(tmp_path):
    # Every other checkpoint here holds either one level or the 199-level scoring grid, so
    # this is the first nine-level one. A grid the store could not round-trip would be
    # found fourteen hours into a run, not at the first save.
    origins = Origins(
        days=tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(1_200)),
        min_train=1_095,
        horizon=HORIZON,
        step=7,
        train_window=1_095,
    )
    path = tmp_path / "TimesFM.npz"
    checkpoint = open_checkpoint(path, NAME, origins, 3, DECILES)
    forecast = np.arange(3 * HORIZON * DECILES.size, dtype=float).reshape(
        3, HORIZON, DECILES.size
    )
    checkpoint.record(0, forecast, 41.5)
    checkpoint.save()

    resumed = open_checkpoint(path, NAME, origins, 3, DECILES)
    assert resumed.n_done == 1
    assert np.array_equal(resumed.quantiles[0], forecast)
    assert resumed.seconds[0] == pytest.approx(41.5)


@pytest.mark.slow
def test_it_forecasts_sorted_non_negative_deciles_for_every_series():
    pytest.importorskip("timesfm")
    context = 512
    model = ZeroShotTimesFM(horizon=HORIZON, context=context)
    day = np.arange(context)
    series = np.stack(
        [level * (1 + 0.1 * np.sin(2 * np.pi * day / 7)) for level in (4_000.0, 20.0, 0.0)]
    )
    deciles = model.forecast(series, HORIZON, DECILES)

    assert deciles.shape == (3, HORIZON, DECILES.size)
    assert (deciles >= 0).all()
    assert (np.diff(deciles, axis=-1) >= 0).all()
    # A weekly shape this clean should put the city's median well above the small area's.
    assert deciles[0, :, MEDIAN_AT].mean() > deciles[1, :, MEDIAN_AT].mean()


@pytest.mark.slow
def test_the_same_window_forecasts_the_same_way_twice():
    # Zero-shot inference with no sampling: a rerun that moved would make every scored
    # number unreproducible, and this is the cheapest place to find that out.
    pytest.importorskip("timesfm")
    context = 512
    model = ZeroShotTimesFM(horizon=HORIZON, context=context)
    series = np.tile(np.linspace(100.0, 200.0, context), (2, 1))
    first = model.forecast(series, HORIZON, DECILES)
    second = model.forecast(series, HORIZON, DECILES)
    assert np.array_equal(first, second)
