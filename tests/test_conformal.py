"""Conformal intervals.

PLAN.md section 4 names one of these among the tests that matter: the adaptive conformal
update tracks a synthetic shift within the expected lag. The look-ahead tests matter just
as much, because a conformal method learns from its own past and is the part of this
project most able to cheat without anyone noticing.
"""

import numpy as np
import pytest

from headroom.conformal.aci import GAMMA_GRID, AdaptiveConformal
from headroom.conformal.agaci import AggregatedConformal, _softmin
from headroom.conformal.split import (
    SplitConformal,
    available_upto,
    conformal_quantile,
    horizon_lag,
    nonconformity,
)

ALPHA = 0.10


def _shifted_scores(n: int = 1_200, at: int = 600, seed: int = 0):
    """Absolute residuals that triple in size partway through, and stay there."""
    rng = np.random.default_rng(seed)
    scores = np.abs(rng.normal(0.0, 10.0, size=n))
    scores[at:] = np.abs(rng.normal(0.0, 30.0, size=n - at))
    return scores


# --- the feedback rule -----------------------------------------------------------


def test_the_outcome_of_a_long_horizon_forecast_is_not_known_at_the_next_origin():
    """With weekly origins, a 14-day-ahead forecast's outcome is two origins away."""
    assert horizon_lag(1, 7) == 1
    assert horizon_lag(7, 7) == 1
    assert horizon_lag(8, 7) == 2
    assert horizon_lag(14, 7) == 2
    assert horizon_lag(15, 7) == 3


def test_available_upto_never_includes_an_outcome_that_has_not_happened():
    # One-day horizon at a weekly step: the previous origin's outcome is known.
    assert available_upto(origin=5, horizon_step=1, origin_step=7) == 5
    # Fourteen days ahead: the previous origin's forecast has not landed yet.
    assert available_upto(origin=5, horizon_step=14, origin_step=7) == 4
    # Nothing at all is known at the first origins.
    assert available_upto(origin=0, horizon_step=1, origin_step=7) == 0
    assert available_upto(origin=1, horizon_step=14, origin_step=7) == 0


def test_a_method_is_blind_to_scores_it_should_not_have_seen():
    """Corrupting only the future must not change any width already produced.

    If this fails, the method is reading ahead and every coverage number is void.
    """
    scores = _shifted_scores()
    corrupted = scores.copy()
    corrupted[700:] = 10_000.0

    for method in (
        SplitConformal(ALPHA),
        AdaptiveConformal(ALPHA, gamma=0.05),
        AggregatedConformal(ALPHA),
    ):
        honest = method.widths(scores, horizon_step=14, origin_step=7)
        tampered = method.widths(corrupted, horizon_step=14, origin_step=7)
        # Outcomes from origin 700 become visible from origin 701 onward, so everything
        # strictly before that must be untouched.
        np.testing.assert_array_equal(honest[:701], tampered[:701])
        assert not np.array_equal(honest[702:], tampered[702:])


def test_no_interval_is_produced_before_anything_has_been_observed():
    widths = SplitConformal(ALPHA).widths(_shifted_scores(), 14, 7)
    assert np.isnan(widths[0])
    assert np.isnan(widths[1])
    assert np.isfinite(widths[2])


# --- the conformal quantile ------------------------------------------------------


def test_the_finite_sample_correction_is_applied_rather_than_a_plain_quantile():
    scores = np.arange(1.0, 11.0)  # n = 10
    # ceil(11 * 0.9) / 10 = 1.0, so ten scores cannot certify 90 percent: widest wins.
    assert conformal_quantile(scores, 0.10) == 10.0
    # With 19 scores, ceil(20 * 0.9) / 19 = 18/19, which is below 1 and is a real quantile.
    assert conformal_quantile(np.arange(1.0, 20.0), 0.10) == 18.0


def test_a_short_calibration_window_starts_conservative_rather_than_undercovering():
    rng = np.random.default_rng(1)
    scores = np.abs(rng.normal(0.0, 10.0, size=500))
    few = conformal_quantile(scores[:8], 0.10)
    many = conformal_quantile(scores, 0.10)
    assert few == scores[:8].max()
    assert few >= many * 0.5  # it does not silently produce a tiny width


def test_the_conformal_quantile_refuses_empty_input_or_a_bad_alpha():
    with pytest.raises(ValueError, match="no calibration"):
        conformal_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_quantile(np.ones(5), 0.0)


def test_nonconformity_is_the_absolute_residual():
    assert nonconformity(np.array([10.0, 5.0]), np.array([7.0, 9.0])).tolist() == [3.0, 4.0]


# --- split conformal covers when its assumption holds, and fails when it does not ---


def test_split_conformal_covers_on_an_exchangeable_sequence():
    """Its guarantee is real; it is the assumption that does not hold on a time series."""
    rng = np.random.default_rng(2)
    scores = np.abs(rng.normal(0.0, 10.0, size=3_000))
    widths = SplitConformal(ALPHA).widths(scores, horizon_step=1, origin_step=7)

    settled = np.isfinite(widths) & (np.arange(scores.size) > 100)
    realised = float((scores[settled] <= widths[settled]).mean())

    assert realised == pytest.approx(1.0 - ALPHA, abs=0.02)
    # The guarantee is one-sided: at least 1 - alpha, never promised to be exactly it.
    assert realised >= 1.0 - ALPHA


def test_split_conformal_collapses_through_a_shift_and_only_recovers_with_the_window():
    """PLAN.md section 9's candidate 1, on a fixture. The real one is the March 2020 chart."""
    at = 600
    scores = _shifted_scores(at=at)
    method = SplitConformal(ALPHA, window=52)
    widths = method.widths(scores, horizon_step=1, origin_step=7)

    def coverage(lo: int, hi: int) -> float:
        window = slice(lo, hi)
        ok = np.isfinite(widths[window])
        return float((scores[window][ok] <= widths[window][ok]).mean())

    before = coverage(300, at)
    during = coverage(at, at + 10)
    after = coverage(at + 200, at + 400)

    assert before == pytest.approx(1.0 - ALPHA, abs=0.05)
    assert during < 0.5  # the interval is calibrated on a world that has gone
    assert after == pytest.approx(1.0 - ALPHA, abs=0.06)

    # It recovers only because the calibration window rolls past the break, not because
    # the method noticed anything. With a 52-origin window that takes about 52 origins.
    assert coverage(at, at + 20) < coverage(at + 40, at + 60)


# --- adaptive conformal ----------------------------------------------------------


def test_adaptive_conformal_tracks_a_synthetic_shift_within_the_expected_lag():
    """The test PLAN.md section 4 names.

    The lag is roughly 1 / gamma updates, so a gamma of 0.1 should be back near nominal
    within a few hundred origins of the break, and comfortably inside 1 / gamma * 10.
    """
    at = 600
    scores = _shifted_scores(n=2_000, at=at)
    method = AdaptiveConformal(ALPHA, gamma=0.1, window=52)
    widths, alphas = method.trace(scores, horizon_step=1, origin_step=7)

    ok = np.isfinite(widths)
    hit = np.zeros(scores.size, dtype=bool)
    hit[ok] = scores[ok] <= widths[ok]

    expected_lag = int(1.0 / method.gamma) * 10  # 100 origins
    recovered = slice(at + expected_lag, scores.size)
    assert float(hit[recovered].mean()) == pytest.approx(1.0 - ALPHA, abs=0.04)

    # It noticed: alpha was driven down by the misses, which is what widened the interval.
    assert np.nanmean(alphas[at : at + expected_lag]) < np.nanmean(alphas[300:at])
    assert np.nanmean(widths[recovered]) > np.nanmean(widths[300:at])


def test_adaptive_conformal_loses_less_coverage_than_split_in_the_weeks_after_a_break():
    """Where the two actually differ, stated honestly.

    Over a long enough record they converge, because split conformal's rolling window
    eventually contains only post-shift data and recalibrates itself. The difference is
    in the weeks immediately after the break, before the window has rolled, and that is
    exactly the period a planner is exposed in.
    """
    at = 600
    scores = _shifted_scores(n=2_000, at=at)
    just_after = slice(at, at + 25)

    def realised(widths: np.ndarray) -> float:
        window = widths[just_after]
        ok = np.isfinite(window)
        return float((scores[just_after][ok] <= window[ok]).mean())

    split = realised(SplitConformal(ALPHA, window=52).widths(scores, 1, 7))
    adaptive = realised(AdaptiveConformal(ALPHA, gamma=0.1, window=52).widths(scores, 1, 7))

    assert adaptive > split
    assert abs(adaptive - (1.0 - ALPHA)) < abs(split - (1.0 - ALPHA))


def test_adaptive_conformal_does_not_claim_per_period_coverage():
    """The honest limitation, asserted rather than only written down.

    It can only respond to a shift after the shift has already cost it coverage, so there
    is always a window right after the break where it is far from nominal.
    """
    at = 600
    scores = _shifted_scores(n=2_000, at=at)
    widths = AdaptiveConformal(ALPHA, gamma=0.01, window=52).widths(scores, 1, 7)

    ok = np.isfinite(widths)
    hit = np.zeros(scores.size, dtype=bool)
    hit[ok] = scores[ok] <= widths[ok]

    assert float(hit[at : at + 10].mean()) < 0.7  # well below nominal, just after the break


def test_a_stable_sequence_leaves_alpha_near_its_target():
    rng = np.random.default_rng(3)
    scores = np.abs(rng.normal(0.0, 10.0, size=2_000))
    _, alphas = AdaptiveConformal(ALPHA, gamma=0.01).trace(scores, 1, 7)
    assert np.nanmean(alphas[500:]) == pytest.approx(ALPHA, abs=0.03)


def test_alpha_stays_inside_the_range_where_the_update_is_reversible():
    scores = np.concatenate([np.full(400, 1.0), np.full(400, 1_000.0)])
    _, alphas = AdaptiveConformal(ALPHA, gamma=0.5).trace(scores, 1, 7)
    finite = alphas[np.isfinite(alphas)]
    assert np.all(finite > 0.0)
    assert np.all(finite < 1.0)


# --- aggregation -----------------------------------------------------------------


def test_aggregation_lands_between_its_experts_rather_than_outside_them():
    scores = _shifted_scores(n=1_500)
    aggregated = AggregatedConformal(ALPHA).widths(scores, 1, 7)
    experts = np.vstack([AdaptiveConformal(ALPHA, g).widths(scores, 1, 7) for g in GAMMA_GRID])

    ok = np.isfinite(aggregated)
    assert np.all(aggregated[ok] >= experts[:, ok].min(axis=0) - 1e-9)
    assert np.all(aggregated[ok] <= experts[:, ok].max(axis=0) + 1e-9)


def test_aggregation_gets_close_to_nominal_without_anyone_choosing_a_step_size():
    """The point of AgACI: no gamma was picked, by hand or with hindsight."""
    scores = _shifted_scores(n=2_500, at=800)
    widths = AggregatedConformal(ALPHA).widths(scores, 1, 7)

    ok = np.isfinite(widths)
    realised = float((scores[ok] <= widths[ok]).mean())
    assert realised == pytest.approx(1.0 - ALPHA, abs=0.05)


def test_the_weights_move_towards_the_experts_that_have_been_doing_well():
    scores = _shifted_scores(n=2_000, at=600)
    _, weights = AggregatedConformal(ALPHA).trace(scores, 1, 7)

    settled = weights[np.isfinite(weights).all(axis=1)]
    assert settled.shape[1] == len(GAMMA_GRID)
    assert np.allclose(settled.sum(axis=1), 1.0)
    # The weights are not still uniform by the end: the data has said something.
    assert settled[-1].max() > 1.5 / len(GAMMA_GRID)


def test_softmin_puts_the_most_weight_on_the_smallest_loss():
    weights = _softmin(np.array([10.0, 20.0, 30.0]), 0.05)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_softmin_survives_losses_large_enough_to_overflow_a_naive_exponential():
    weights = _softmin(np.array([1e6, 1e6 + 10.0]), 0.05)
    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1]
