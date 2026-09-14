"""MinT reconciliation: coherent at every origin, equal to the library, blind to the future.

PLAN.md makes coherence a test rather than a claim. The first test here is that claim for
arbitrary inputs; the library test pins the arithmetic to hierarchicalforecast's own.
"""

import numpy as np
import pytest

from headroom.conformal.split import available_upto
from headroom.hierarchy.spec import Hierarchy
from headroom.reconcile.mint import (
    mint_matrix,
    reconcile_backtest,
    shift_quantiles,
    shrunk_covariance,
)

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


def _errors(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    leaf = rng.normal(size=(n, 4)) * np.array([1.0, 2.0, 0.5, 1.5]) + rng.normal(size=(n, 1))
    return np.asarray(leaf @ S.T + rng.normal(size=(n, 7)) * 0.3, dtype=np.float64)


def test_reconciled_forecasts_are_coherent_whatever_the_input():
    rng = np.random.default_rng(1)
    w, _ = shrunk_covariance(_errors())
    projection = S @ mint_matrix(S, w)
    for _ in range(50):
        base = rng.normal(100.0, 30.0, size=7)  # incoherent on purpose
        assert HIERARCHY.coherence_error(projection @ base) < 1e-9


def test_a_coherent_forecast_comes_back_unchanged():
    w, _ = shrunk_covariance(_errors())
    coherent = S @ np.array([20.0, 35.0, 12.0, 40.0])
    np.testing.assert_allclose(S @ mint_matrix(S, w) @ coherent, coherent, atol=1e-9)


def test_identity_weights_give_ordinary_least_squares():
    ols = S @ np.linalg.inv(S.T @ S) @ S.T
    np.testing.assert_allclose(S @ mint_matrix(S, np.eye(7)), ols, atol=1e-12)


def test_the_projection_equals_hierarchicalforecasts_mint_shrink():
    methods = pytest.importorskip("hierarchicalforecast.methods")
    errors = _errors()
    y_insample = np.random.default_rng(2).normal(size=(7, errors.shape[0]))
    p_library, w_library = methods.MinTrace(method="mint_shrink")._get_PW_matrices(
        S=S,
        y_hat=np.ones((7, 3)),
        y_insample=y_insample,
        y_hat_insample=y_insample - errors.T,
    )
    w, _ = shrunk_covariance(errors)
    np.testing.assert_allclose(w, w_library, atol=1e-6)
    np.testing.assert_allclose(mint_matrix(S, w), p_library, atol=1e-6)


def test_shrinkage_intensity_is_high_for_noise_and_low_for_strong_correlation():
    rng = np.random.default_rng(3)
    _, noise = shrunk_covariance(rng.normal(size=(30, 7)))
    common = rng.normal(size=(200, 1))
    _, correlated = shrunk_covariance(common + rng.normal(size=(200, 7)) * 0.05)
    assert 0.0 <= correlated < 0.05
    assert noise > 0.5


def test_errors_that_never_vary_are_refused():
    with pytest.raises(ValueError, match="never vary"):
        shrunk_covariance(np.zeros((10, 3)))


def _backtest(n_origins: int = 120, seed: int = 4):
    rng = np.random.default_rng(seed)
    horizon = 14
    leaves = rng.normal(100.0, 10.0, size=(n_origins, 4, horizon))
    actual = np.einsum("nl,olh->onh", S, leaves)
    median = actual + np.einsum("nl,olh->onh", S, rng.normal(size=(n_origins, 4, horizon)) * 5)
    median += rng.normal(size=median.shape) * 3  # base forecasts that do not sum
    return median, actual


def test_every_reconciled_origin_is_coherent_and_earlier_ones_are_not_reconciled():
    median, actual = _backtest()
    result = reconcile_backtest(median, actual, HIERARCHY, origin_step=7)
    assert result.first_valid == 53
    assert np.all(np.isnan(result.medians[:53]))
    assert result.coherence_error < 1e-8
    for origin in range(53, median.shape[0]):
        for step in range(14):
            assert HIERARCHY.coherence_error(result.medians[origin, :, step]) < 1e-8


def test_an_outcome_not_yet_observed_cannot_change_a_reconciliation():
    median, actual = _backtest()
    base = reconcile_backtest(median, actual, HIERARCHY, origin_step=7).medians
    origin = 90
    for step in (1, 8, 14):
        end = available_upto(origin, step, 7)
        corrupted = actual.copy()
        corrupted[end:, :, step - 1] += 1e6
        after = reconcile_backtest(median, corrupted, HIERARCHY, origin_step=7).medians
        np.testing.assert_array_equal(base[origin, :, step - 1], after[origin, :, step - 1])


def test_quantiles_move_with_their_median_and_keep_their_spread():
    quantiles = np.array([[80.0, 100.0, 120.0], [1.0, 3.0, 5.0]])
    base = np.array([100.0, 3.0])
    reconciled = np.array([104.0, 1.5])
    shifted = shift_quantiles(quantiles, base, reconciled)
    np.testing.assert_allclose(shifted[0], [84.0, 104.0, 124.0])
    np.testing.assert_allclose(shifted[1], [0.0, 1.5, 3.5])  # the floor binds below zero
