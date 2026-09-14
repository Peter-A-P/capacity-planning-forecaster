"""The conformal predictive distribution for a model that forecasts only a median.

The feedback test is the one that matters: a distribution built from past errors is the
easiest place in the project to calibrate on an error whose outcome had not happened yet.
"""

import numpy as np
import pytest

from headroom.conformal.predictive import (
    first_valid_origin,
    order_statistic_ranks,
    predictive_quantiles,
)
from headroom.conformal.split import available_upto
from headroom.score import levels as lv

STEP = 7
HORIZON = 14
WINDOW = 52


def test_ranks_follow_the_conformal_order_statistic():
    ranks = order_statistic_ranks(np.array([0.01, 0.05, 0.5, 0.95, 0.999]), 52)
    # k = ceil(53 q): 1, 3, 27, 51, and 53 clipped to 52; zero-based below.
    np.testing.assert_array_equal(ranks, [0, 2, 26, 50, 51])


def test_the_first_origin_is_where_the_longest_step_has_a_full_window():
    first = first_valid_origin(HORIZON, STEP, WINDOW)
    assert available_upto(first, HORIZON, STEP) == WINDOW
    assert available_upto(first - 1, HORIZON, STEP) < WINDOW
    assert first == 53


def _fixture(n_origins: int = 400, n_nodes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    point = np.full((n_origins, n_nodes, HORIZON), 1000.0)
    actual = point + rng.normal(0.0, 50.0, size=point.shape)
    return point, actual


def test_origins_before_a_full_window_have_no_distribution_and_later_ones_all_do():
    point, actual = _fixture()
    result = predictive_quantiles(point, actual, lv.REPORTING, STEP, WINDOW)
    assert result.first_valid == 53
    assert np.all(np.isnan(result.quantiles[:53]))
    assert np.all(np.isfinite(result.quantiles[53:]))


def test_an_error_not_yet_observed_cannot_change_a_distribution():
    point, actual = _fixture()
    levels = lv.REPORTING
    base = predictive_quantiles(point, actual, levels, STEP, WINDOW).quantiles

    origin = 200
    for step in (1, 7, 8, 14):
        end = available_upto(origin, step, STEP)
        corrupted = actual.copy()
        corrupted[end:, :, step - 1] = 1e9  # every outcome not yet known at this origin
        after = predictive_quantiles(point, corrupted, levels, STEP, WINDOW).quantiles
        np.testing.assert_array_equal(base[origin, :, step - 1], after[origin, :, step - 1])

        corrupted = actual.copy()
        corrupted[end - 1, :, step - 1] = 1e9  # the most recent known outcome does count
        after = predictive_quantiles(point, corrupted, levels, STEP, WINDOW).quantiles
        assert not np.array_equal(base[origin, :, step - 1], after[origin, :, step - 1])


def test_exchangeable_errors_are_covered_at_about_the_nominal_rate():
    point, actual = _fixture(n_origins=2_000, n_nodes=5, seed=3)
    levels = np.array([0.05, 0.5, 0.95])
    q = predictive_quantiles(point, actual, levels, STEP, WINDOW).quantiles[53:]
    covered = (actual[53:] >= q[..., 0]) & (actual[53:] <= q[..., 2])
    # Conformal with k = ceil((n + 1) q) at n = 52 certifies at least 49/53, about 0.925.
    assert 0.90 <= covered.mean() <= 0.95


def test_quantiles_are_ordered_and_floored_at_zero():
    point, actual = _fixture()
    point[:] = 5.0  # a low-volume series whose errors reach below zero
    result = predictive_quantiles(point, actual, lv.SCORING, STEP, WINDOW).quantiles[53:]
    assert np.all(result >= 0.0)
    assert np.all(np.diff(result, axis=-1) >= 0.0)


def test_a_backtest_too_short_for_any_window_is_refused():
    point, actual = _fixture(n_origins=40)
    with pytest.raises(ValueError, match="too few"):
        predictive_quantiles(point, actual, lv.REPORTING, STEP, WINDOW)
