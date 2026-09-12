"""Probabilistic scores.

PLAN.md section 4 names two of these among the tests that matter: CRPS from quantiles
against a closed form on a normal fixture, and the block bootstrap respecting origin
ordering.
"""

import numpy as np
import pytest
from scipy.stats import norm

from headroom.score import levels as lv
from headroom.score.bootstrap import (
    block_indices,
    confidence_interval,
    skill_interval,
)
from headroom.score.coverage import (
    covered,
    empirical_coverage,
    interval_bounds,
    level_index,
    rolling_coverage,
    worst_window,
)
from headroom.score.crps import (
    crossing_rate,
    crps_from_quantiles,
    crps_normal,
    sort_quantiles,
)
from headroom.score.pinball import pinball_by_level, pinball_loss
from headroom.score.skill import paired_difference, skill
from headroom.score.width import relative_width, width

MU, SD = 100.0, 15.0


def _normal_quantiles(levels: np.ndarray, n: int, mu: float = MU, sd: float = SD) -> np.ndarray:
    return np.broadcast_to(norm.ppf(levels, mu, sd), (n, levels.size)).copy()


# --- pinball ---------------------------------------------------------------------


def test_the_pinball_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = MU + SD * rng.normal(size=200_000)

    for level in (0.1, 0.5, 0.9):
        truth = float(norm.ppf(level, MU, SD))
        at_truth = pinball_loss(np.float64(truth), y, level).mean()
        for offset in (-3.0, -1.0, 1.0, 3.0):
            assert pinball_loss(np.float64(truth + offset), y, level).mean() > at_truth


def test_the_pinball_loss_at_the_median_is_half_the_absolute_error():
    forecast = np.array([10.0, 20.0])
    observed = np.array([13.0, 14.0])
    assert pinball_loss(forecast, observed, 0.5) == pytest.approx([1.5, 3.0])


def test_a_high_quantile_charges_far_more_for_being_under_than_over():
    """At 0.95 an under-forecast costs nineteen times an over-forecast of the same size."""
    under = pinball_loss(np.float64(100.0), np.float64(110.0), 0.95)
    over = pinball_loss(np.float64(100.0), np.float64(90.0), 0.95)
    assert under / over == pytest.approx(19.0)


def test_pinball_by_level_matches_the_scalar_version_at_every_level():
    quantiles = _normal_quantiles(lv.REPORTING, 5)
    observed = np.array([90.0, 100.0, 110.0, 130.0, 70.0])
    table = pinball_by_level(quantiles, observed, lv.REPORTING)

    for j, level in enumerate(lv.REPORTING):
        assert table[:, j] == pytest.approx(
            pinball_loss(quantiles[:, j], observed, float(level))
        )


def test_a_level_outside_the_open_unit_interval_is_refused():
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="level must be in"):
            pinball_loss(np.float64(1.0), np.float64(1.0), bad)


def test_a_quantile_grid_that_is_not_a_grid_is_refused():
    observed = np.zeros(3)
    quantiles = np.zeros((3, 3))
    for bad, message in [
        (np.array([0.5, 0.2, 0.9]), "strictly increasing"),
        (np.array([0.0, 0.5, 0.9]), "strictly inside"),
        (np.array([0.1, 0.5, 1.0]), "strictly inside"),
    ]:
        with pytest.raises(ValueError, match=message):
            pinball_by_level(quantiles, observed, bad)


# --- CRPS ------------------------------------------------------------------------


def test_crps_from_quantiles_matches_the_closed_form_on_a_normal():
    """The test PLAN.md section 4 names. Tolerance is the measured quadrature error."""
    rng = np.random.default_rng(1)
    observed = MU + SD * rng.normal(size=20_000)
    quantiles = _normal_quantiles(lv.SCORING, observed.size)

    approx = crps_from_quantiles(quantiles, observed, lv.SCORING).mean()
    exact = crps_normal(MU, SD, observed).mean()

    assert approx == pytest.approx(exact, rel=1e-4)


def test_the_quadrature_error_is_downward_and_shrinks_as_the_grid_is_refined():
    """The docstring of headroom.score.crps claims both. This is the claim, measured."""
    rng = np.random.default_rng(2)
    observed = MU + SD * rng.normal(size=5_000)
    exact = crps_normal(MU, SD, observed).mean()

    errors = []
    for size in (49, 199, 999):
        grid = np.linspace(1.0 / (2 * size), 1.0 - 1.0 / (2 * size), size)
        approx = crps_from_quantiles(_normal_quantiles(grid, observed.size), observed, grid)
        errors.append(approx.mean() - exact)

    assert all(e < 0.0 for e in errors)
    assert abs(errors[0]) > abs(errors[1]) > abs(errors[2])


def test_the_reporting_grid_is_far_too_coarse_to_integrate_crps_on():
    """Which is why headroom.score.levels keeps a separate, denser scoring grid."""
    rng = np.random.default_rng(3)
    observed = MU + SD * rng.normal(size=5_000)
    exact = crps_normal(MU, SD, observed).mean()

    coarse = crps_from_quantiles(
        _normal_quantiles(lv.REPORTING, observed.size), observed, lv.REPORTING
    ).mean()

    assert coarse < exact
    assert abs(coarse / exact - 1.0) > 0.02


def test_crps_of_a_point_forecast_is_the_absolute_error():
    """A degenerate distribution scores exactly |y - x|, which anchors the units."""
    observed = np.array([100.0, 100.0])
    point = np.array([90.0, 130.0])
    quantiles = np.repeat(point[:, np.newaxis], lv.SCORING.size, axis=1)

    assert crps_from_quantiles(quantiles, observed, lv.SCORING) == pytest.approx(
        np.abs(observed - point)
    )


def test_crps_prefers_the_sharper_of_two_calibrated_forecasts():
    """Coverage cannot tell these apart; CRPS charges the wider one for its width."""
    rng = np.random.default_rng(4)
    observed = MU + SD * rng.normal(size=20_000)

    honest = crps_from_quantiles(
        _normal_quantiles(lv.SCORING, observed.size, sd=SD), observed, lv.SCORING
    ).mean()
    hedged = crps_from_quantiles(
        _normal_quantiles(lv.SCORING, observed.size, sd=3 * SD), observed, lv.SCORING
    ).mean()

    assert honest < hedged


def test_crps_normal_refuses_a_non_positive_standard_deviation():
    with pytest.raises(ValueError, match="strictly positive"):
        crps_normal(0.0, 0.0, 1.0)


def test_crossed_quantiles_are_counted_and_then_sorted():
    quantiles = np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0], [5.0, 4.0, 6.0], [0.0, 1.0, 2.0]])
    assert crossing_rate(quantiles) == pytest.approx(0.5)

    fixed = sort_quantiles(quantiles)
    assert crossing_rate(fixed) == 0.0
    assert np.array_equal(fixed[0], np.array([1.0, 2.0, 3.0]))


def test_sorting_crossed_quantiles_never_makes_the_score_worse():
    rng = np.random.default_rng(5)
    observed = MU + SD * rng.normal(size=2_000)
    quantiles = _normal_quantiles(lv.SCORING, observed.size)
    quantiles += rng.normal(0.0, 4.0, size=quantiles.shape)  # enough noise to cross

    assert crossing_rate(quantiles) > 0.0
    crossed = crps_from_quantiles(quantiles, observed, lv.SCORING).mean()
    sorted_ = crps_from_quantiles(sort_quantiles(quantiles), observed, lv.SCORING).mean()
    assert sorted_ <= crossed


# --- coverage and width ----------------------------------------------------------


def test_a_calibrated_forecast_covers_at_its_nominal_rate():
    rng = np.random.default_rng(6)
    observed = MU + SD * rng.normal(size=50_000)
    quantiles = _normal_quantiles(lv.REPORTING, observed.size)

    for nominal in lv.NOMINAL_COVERAGES:
        lower, upper = interval_bounds(quantiles, lv.REPORTING, nominal)
        assert empirical_coverage(covered(lower, upper, observed)) == pytest.approx(
            nominal, abs=0.01
        )


def test_a_too_narrow_forecast_undercovers_and_is_narrower():
    rng = np.random.default_rng(7)
    observed = MU + SD * rng.normal(size=20_000)
    honest = _normal_quantiles(lv.REPORTING, observed.size, sd=SD)
    narrow = _normal_quantiles(lv.REPORTING, observed.size, sd=SD / 2)

    honest_hits = covered(*interval_bounds(honest, lv.REPORTING, 0.90), observed)
    narrow_hits = covered(*interval_bounds(narrow, lv.REPORTING, 0.90), observed)

    assert empirical_coverage(narrow_hits) < 0.75
    assert empirical_coverage(honest_hits) == pytest.approx(0.90, abs=0.01)
    assert width(narrow, lv.REPORTING, 0.90).mean() < width(honest, lv.REPORTING, 0.90).mean()


def test_an_interval_bound_must_be_in_the_grid_rather_than_interpolated():
    grid = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="not in the quantile grid"):
        interval_bounds(np.zeros((2, 3)), grid, 0.95)
    assert level_index(grid, 0.9) == 2


def test_the_interval_is_closed_so_a_bound_landing_on_the_count_is_a_hit():
    lower = np.array([10.0, 10.0, 10.0])
    upper = np.array([20.0, 20.0, 20.0])
    observed = np.array([10.0, 20.0, 20.5])
    assert covered(lower, upper, observed).tolist() == [True, True, False]


def test_rolling_coverage_is_trailing_and_refuses_to_report_a_short_window():
    hits = np.array([True] * 10 + [False] * 10)
    rolling = rolling_coverage(hits, window=5)

    assert np.all(np.isnan(rolling[:4]))
    assert rolling[4] == pytest.approx(1.0)
    assert rolling[9] == pytest.approx(1.0)  # trailing: still all hits
    assert rolling[14] == pytest.approx(0.0)  # the window has moved fully past the break
    assert rolling.size == hits.size


def test_rolling_coverage_finds_a_break_that_the_whole_period_average_hides():
    """The reason the headline chart is a rolling window and not one number."""
    rng = np.random.default_rng(8)
    hits = rng.random(4_000) < 0.925  # comfortably over nominal for most of the record
    hits[2_000:2_090] = False  # ninety origins where the interval failed completely

    assert empirical_coverage(hits) > 0.88  # the single number still looks close to 0.90
    _, worst = worst_window(hits, window=90)
    assert worst == pytest.approx(0.0)


def test_rolling_coverage_refuses_a_window_longer_than_the_record():
    with pytest.raises(ValueError, match="longer than"):
        rolling_coverage(np.ones(10, dtype=bool), window=11)


def test_relative_width_makes_a_leaf_and_the_city_comparable():
    levels = np.array([0.05, 0.5, 0.95])
    city = np.array([[3_610.0, 3_800.0, 3_990.0]])
    leaf = np.array([[19.0, 20.0, 21.0]])

    assert width(city, levels, 0.90)[0] == pytest.approx(380.0)
    assert width(leaf, levels, 0.90)[0] == pytest.approx(2.0)
    # A hundred and ninety times the absolute width, the same width relative to the series.
    assert relative_width(city, levels, 0.90, np.array([3_800.0]))[0] == pytest.approx(0.1)
    assert relative_width(leaf, levels, 0.90, np.array([20.0]))[0] == pytest.approx(0.1)


def test_relative_width_refuses_a_series_with_no_scale_to_divide_by():
    levels = np.array([0.05, 0.5, 0.95])
    with pytest.raises(ValueError, match="strictly positive"):
        relative_width(np.array([[0.0, 1.0, 2.0]]), levels, 0.90, np.array([0.0]))


# --- skill -----------------------------------------------------------------------


def test_skill_is_positive_when_the_method_beats_the_baseline():
    assert skill(88.0, 100.0) == pytest.approx(0.12)
    assert skill(100.0, 100.0) == pytest.approx(0.0)
    assert skill(150.0, 100.0) == pytest.approx(-0.5)


def test_skill_against_a_perfect_baseline_is_refused_rather_than_infinite():
    with pytest.raises(ValueError, match="undefined"):
        skill(1.0, 0.0)


def test_paired_difference_is_positive_when_the_first_method_is_better():
    score = np.array([10.0, 12.0])
    other = np.array([11.0, 11.0])
    assert paired_difference(score, other).tolist() == [1.0, -1.0]


def test_unpaired_scores_are_refused():
    with pytest.raises(ValueError, match="same shape"):
        paired_difference(np.zeros(3), np.zeros(4))


# --- the block bootstrap ---------------------------------------------------------


def test_the_block_bootstrap_draws_contiguous_increasing_runs_of_origins():
    """The test PLAN.md section 4 names.

    A bug that shuffled inside a block would turn this back into the independent
    bootstrap and narrow every interval in the project without failing anything else.
    """
    rng = np.random.default_rng(9)
    block = 7
    for _ in range(200):
        idx = block_indices(100, block, rng)
        assert idx.size == 100
        assert idx.min() >= 0
        assert idx.max() < 100
        # Every whole block is a run of consecutive, increasing indices.
        for start in range(0, idx.size - block + 1, block):
            run = idx[start : start + block]
            assert np.array_equal(np.diff(run), np.ones(block - 1, dtype=idx.dtype))


def test_a_block_longer_than_the_record_is_refused():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="block must be between"):
        block_indices(10, 11, rng)


def test_the_block_bootstrap_gives_a_wider_interval_than_ignoring_dependence():
    """The reason for blocks at all: independent resampling reports false precision."""
    rng = np.random.default_rng(11)
    # A strongly autocorrelated score series, as a method's scores through a regime are.
    noise = rng.normal(size=2_000)
    values = np.convolve(noise, np.ones(60) / 60.0, mode="same") + 10.0

    _, lo_block, hi_block = confidence_interval(values, block=28, seed=0)
    _, lo_iid, hi_iid = confidence_interval(values, block=1, seed=0)

    assert (hi_block - lo_block) > 2.0 * (hi_iid - lo_iid)


def test_a_bootstrap_interval_brackets_the_point_estimate_and_reproduces():
    rng = np.random.default_rng(12)
    values = rng.normal(10.0, 2.0, size=500)

    point, lower, upper = confidence_interval(values, seed=3)
    assert lower < point < upper
    assert (point, lower, upper) == confidence_interval(values, seed=3)


def test_pairing_the_skill_bootstrap_tightens_the_interval_it_reports():
    """Both methods resampled on the same origins, so origin variation cancels."""
    rng = np.random.default_rng(13)
    origin_effect = np.abs(np.cumsum(rng.normal(size=600))) + 50.0  # huge, shared
    baseline = origin_effect + rng.normal(0.0, 1.0, size=600)
    score = 0.9 * origin_effect + rng.normal(0.0, 1.0, size=600)  # 10 percent better

    point, lower, upper = skill_interval(score, baseline, seed=0)

    assert point == pytest.approx(0.10, abs=0.01)
    assert lower < point < upper
    assert (upper - lower) < 0.05  # the shared origin effect has cancelled


def test_the_skill_bootstrap_refuses_score_series_that_are_not_paired():
    with pytest.raises(ValueError, match="same origins"):
        skill_interval(np.ones(10), np.ones(9))


def test_an_observation_outside_the_grid_is_charged_for_the_whole_miss():
    """The flat-tail terms exist for this case, which is what a shift produces.

    Padding the integrand to zero at the ends instead, which is the obvious
    implementation, understates the score by about half a percent here, and only ever for
    the forecasts that missed.
    """
    quantiles = _normal_quantiles(lv.SCORING, 1)
    for observation in (300.0, 1_000.0):
        approx = crps_from_quantiles(quantiles, np.array([observation]), lv.SCORING)[0]
        exact = float(crps_normal(MU, SD, observation))
        assert approx == pytest.approx(exact, rel=1e-3)
        # The flat tail is conservative: never below what the true distribution scores.
        assert approx >= exact
