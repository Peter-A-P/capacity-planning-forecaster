"""Probabilistic reconciliation: every path coherent, and the marginals deliberately not.

PLAN.md section 2.5 asks for coherence verified numerically at every origin. The first test
is that claim over every path rather than only the median. The one after it is the opposite
claim, and it is here so that nobody later "fixes" the quantiles into summing, which would
be false about any hierarchy whose parts do not peak together.
"""

import numpy as np
import pytest

from headroom.conformal.split import WINDOW, available_upto
from headroom.hierarchy.spec import Hierarchy
from headroom.reconcile.mint import reconcile_backtest
from headroom.reconcile.paths import reconcile_paths

S = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)
HIERARCHY = Hierarchy(
    nodes=("NYC", "NYC/A", "NYC/B", "NYC/A/1", "NYC/A/2", "NYC/B/1", "NYC/B/2"),
    leaves=("NYC/A/1", "NYC/A/2", "NYC/B/1", "NYC/B/2"),
    levels=("city", "borough", "borough", "area", "area", "area", "area"),
    s_matrix=S,
)
LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
HORIZON = 3
STEP = 7


def _backtest(n_origins: int = 120, seed: int = 0, scale: float = 1.0):
    """A base median and an outcome, incoherent on purpose, shaped like the real thing."""
    rng = np.random.default_rng(seed)
    leaf_truth = rng.gamma(20.0, 2.0, size=(n_origins, 4, HORIZON))
    actual = np.einsum("nl,olh->onh", S, leaf_truth)
    # Base forecasts made one series at a time, so they do not sum.
    median = actual + rng.normal(0.0, 3.0 * scale, size=actual.shape)
    return median, actual


def test_every_path_is_coherent_at_every_origin():
    median, actual = _backtest()
    result = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    assert result.coherence_error < 1e-8


def test_the_base_forecasts_were_not_coherent_to_begin_with():
    # Without this the coherence test above could pass on an input that never needed
    # reconciling, which would make it a test of nothing.
    median, _ = _backtest()
    breach = max(
        HIERARCHY.coherence_error(median[origin, :, step])
        for origin in range(0, median.shape[0], 17)
        for step in range(HORIZON)
    )
    assert breach > 1.0


def test_the_marginal_quantiles_do_not_sum_and_are_not_meant_to():
    # A coherent distribution does not have coherent quantiles: the boroughs do not have
    # their bad days together, so the city's 90th percentile is below their sum. If this
    # ever fails, something has replaced the paths with per-node arithmetic.
    median, actual = _backtest()
    result = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    upper = list(LEVELS).index(0.9)
    city = result.quantiles[result.first_valid :, 0, :, upper]
    boroughs = result.quantiles[result.first_valid :, 1:3, :, upper].sum(axis=1)
    assert (city < boroughs).mean() > 0.9


def test_the_median_of_the_paths_is_coherent_where_the_quantiles_are_not():
    median, actual = _backtest()
    result = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    at = result.first_valid
    assert HIERARCHY.coherence_error(result.medians[at]) < 1e-8


def test_the_point_reconciliation_matches_the_one_mint_reports():
    # The same pass returns both, so a caller needs one call rather than two. If they ever
    # disagreed, the report's point and probabilistic rows would be about different models.
    median, actual = _backtest()
    paths = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    point = reconcile_backtest(median, actual, HIERARCHY, STEP)
    assert paths.first_valid == point.first_valid
    np.testing.assert_allclose(
        paths.medians[paths.first_valid :], point.medians[point.first_valid :], atol=1e-9
    )


def test_the_distribution_is_ordered_and_centred_near_the_reconciled_median():
    median, actual = _backtest()
    result = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    live = result.quantiles[result.first_valid :]
    assert np.isfinite(live).all()
    assert (np.diff(live, axis=-1) >= 0).all()
    middle = list(LEVELS).index(0.5)
    spread = live[..., -1] - live[..., 0]
    assert np.abs(live[..., middle] - result.medians[result.first_valid :]).mean() < float(
        spread.mean()
    )


def test_a_forecast_never_uses_an_error_whose_outcome_had_not_happened():
    # The feedback rule, again, because this module reads the error window itself rather
    # than borrowing one. Corrupt everything from the last usable origin onward and no
    # distribution already produced may move.
    median, actual = _backtest()
    before = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    origin = before.first_valid
    end = available_upto(origin, HORIZON, STEP)
    poisoned = actual.copy()
    poisoned[end:] += 500.0
    after = reconcile_paths(median, poisoned, HIERARCHY, LEVELS, STEP)
    np.testing.assert_allclose(
        before.quantiles[origin], after.quantiles[origin], rtol=0, atol=1e-9
    )


def test_it_reports_what_not_flooring_the_paths_costs():
    # Clipping at zero would break coherence, so the paths are left alone and the share
    # below zero is measured. On a panel whose smallest series sits near zero it is not
    # zero, and the number belongs in the write-up rather than in nobody's hands.
    median, actual = _backtest(scale=6.0)
    result = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    assert 0.0 <= result.negative_share < 0.5


def test_the_window_is_the_one_the_conformal_distribution_uses():
    median, actual = _backtest()
    result = reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
    assert available_upto(result.first_valid, HORIZON, STEP) == WINDOW


def test_shapes_and_a_hierarchy_that_does_not_match_are_refused():
    median, actual = _backtest()
    with pytest.raises(ValueError, match="must match"):
        reconcile_paths(median, actual[:-1], HIERARCHY, LEVELS, STEP)
    with pytest.raises(ValueError, match="nodes"):
        reconcile_paths(median[:, :-1], actual[:, :-1], HIERARCHY, LEVELS, STEP)


def test_a_backtest_too_short_for_one_full_window_is_refused():
    median, actual = _backtest(n_origins=10)
    with pytest.raises(ValueError, match="too few"):
        reconcile_paths(median, actual, HIERARCHY, LEVELS, STEP)
