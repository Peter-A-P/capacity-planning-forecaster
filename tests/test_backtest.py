"""Forecast origins and the seasonal naive baseline.

The look-ahead test is the one that matters most in this file. A backtest that lets a
forecaster see one day it should not is worthless and says nothing about it in its output.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from headroom.backtest.origins import Origin, Origins
from headroom.models.baselines import SEASON, SeasonalNaive, seasonal_lag
from headroom.score import levels as lv

DAYS = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(2_000))


def _origins(**kwargs) -> Origins:
    settings: dict[str, object] = {"min_train": 400, "horizon": 14, "step": 7}
    settings.update(kwargs)
    return Origins(days=DAYS, **settings)  # type: ignore[arg-type]


# --- origins ---------------------------------------------------------------------


def test_no_origin_can_see_a_day_it_is_scored_on():
    """The look-ahead test. Everything this project reports depends on it."""
    schedule = _origins()
    assert len(list(schedule)) > 0

    for origin in schedule:
        train = range(*origin.train.indices(len(DAYS)))
        target = origin.target_days

        assert max(train) == origin.index  # the origin day itself is observed
        assert min(target) == origin.index + 1  # scoring starts the day after
        assert set(train).isdisjoint(target)
        assert DAYS[origin.index] == origin.day


def test_every_origin_is_scored_on_a_full_horizon():
    schedule = _origins()
    for origin in schedule:
        assert len(origin.target_days) == schedule.horizon
        assert max(origin.target_days) < len(DAYS)


def test_the_schedule_stops_before_the_horizon_would_run_off_the_end():
    schedule = _origins()
    last = list(schedule)[-1]

    # The last origin's whole horizon is inside the record, and one step further would
    # not be. The last origin is the last one *on the step grid* that fits, which is not
    # in general the last day that would fit.
    assert max(last.target_days) <= len(DAYS) - 1
    assert last.index + schedule.step + schedule.horizon > len(DAYS) - 1
    assert last.index <= schedule.last_index


def test_origins_are_spaced_by_the_step_and_start_after_the_training_history():
    schedule = _origins(min_train=400, step=7)
    origins = list(schedule)

    assert origins[0].index == 399
    assert [o.index for o in origins[:4]] == [399, 406, 413, 420]
    assert [o.number for o in origins[:4]] == [0, 1, 2, 3]
    assert len(schedule) == len(origins)


def test_refits_fall_on_the_schedule_and_are_counted():
    schedule = _origins(refit_every=4)
    flags = [o.refit for o in schedule]

    assert flags[:8] == [True, False, False, False, True, False, False, False]
    assert sum(flags) == schedule.n_refits


def test_refitting_at_every_origin_is_expressible():
    schedule = _origins(refit_every=1)
    assert all(o.refit for o in schedule)
    assert schedule.n_refits == len(schedule)


def test_a_record_too_short_for_the_schedule_is_refused():
    short = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(100))
    with pytest.raises(ValueError, match="too short"):
        Origins(days=short, min_train=90, horizon=14)


def test_a_non_positive_schedule_parameter_is_refused():
    for bad in ({"horizon": 0}, {"step": 0}, {"refit_every": 0}, {"min_train": 0}):
        with pytest.raises(ValueError, match="must be at least 1"):
            _origins(**bad)


def test_covering_finds_the_forecasts_that_were_in_flight_on_a_day():
    schedule = _origins()
    origins = list(schedule)
    target_day = DAYS[origins[10].index + 3]

    numbers = schedule.covering(target_day)

    assert origins[10].number in numbers
    for number in numbers:
        assert target_day in [DAYS[i] for i in origins[number].target_days]
    assert schedule.covering(date(1999, 1, 1)) == []


def test_an_origins_slices_are_consistent_with_its_index():
    origin = Origin(number=3, index=100, day=DAYS[100], horizon=5, refit=True)
    assert origin.train == slice(0, 101)
    assert origin.target == slice(101, 106)
    assert list(origin.target_days) == [101, 102, 103, 104, 105]


# --- seasonal naive --------------------------------------------------------------


def test_the_seasonal_lag_reaches_back_a_whole_number_of_weeks():
    assert [seasonal_lag(h) for h in range(1, 16)] == [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        21,
    ]
    with pytest.raises(ValueError, match="counts from 1"):
        seasonal_lag(0)


def test_the_point_forecast_is_the_most_recent_same_weekday():
    """A perfectly periodic series must be forecast exactly, at every horizon step."""
    pattern = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    train = np.tile(pattern, (1, 40))
    forecast = SeasonalNaive().forecast(train, horizon=14, levels=lv.REPORTING)

    median = forecast[0, :, lv.REPORTING.tolist().index(0.5)]
    assert median.tolist() == pytest.approx(np.tile(pattern, 2).tolist())


def test_a_periodic_series_gets_a_zero_width_interval_because_it_never_erred():
    pattern = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    train = np.tile(pattern, (1, 40))
    forecast = SeasonalNaive().forecast(train, horizon=7, levels=lv.REPORTING)

    assert np.ptp(forecast[0, 0, :]) == pytest.approx(0.0)


def test_the_interval_widens_with_the_horizon():
    """A fortnight ahead is more uncertain than a day ahead, and the baseline knows it."""
    rng = np.random.default_rng(0)
    weekly = np.tile(np.array([80.0, 100.0, 100.0, 100.0, 100.0, 120.0, 90.0]), 200)
    train = (weekly + np.cumsum(rng.normal(0.0, 2.0, size=weekly.size)))[np.newaxis, :]

    forecast = SeasonalNaive().forecast(train, horizon=14, levels=lv.REPORTING)
    lo = lv.REPORTING.tolist().index(0.05)
    hi = lv.REPORTING.tolist().index(0.95)
    widths = forecast[0, :, hi] - forecast[0, :, lo]

    assert widths[13] > widths[0]


def test_the_forecast_quantiles_never_cross_and_never_go_negative():
    rng = np.random.default_rng(1)
    train = rng.poisson(20.0, size=(5, 500)).astype(np.float64)
    forecast = SeasonalNaive().forecast(train, horizon=14, levels=lv.SCORING)

    assert np.all(np.diff(forecast, axis=-1) >= 0.0)
    assert np.all(forecast >= 0.0)


def test_a_quiet_series_is_floored_at_zero_rather_than_forecasting_negative_arrivals():
    train = np.zeros((1, 200))
    train[0, ::7] = 1.0
    forecast = SeasonalNaive().forecast(train, horizon=14, levels=lv.REPORTING)
    assert np.all(forecast >= 0.0)


def test_the_baseline_is_calibrated_on_a_series_that_obeys_its_own_assumption():
    """Weekly pattern plus independent noise is exactly what seasonal naive assumes."""
    from headroom.score.coverage import covered, empirical_coverage, interval_bounds

    rng = np.random.default_rng(2)
    pattern = np.tile(np.array([80.0, 100.0, 100.0, 100.0, 100.0, 120.0, 90.0]), 300)
    series = pattern + rng.normal(0.0, 8.0, size=pattern.size)

    train = series[np.newaxis, :1_400]
    actual = series[1_400:1_414]
    forecast = SeasonalNaive().forecast(train, horizon=14, levels=lv.REPORTING)

    lower, upper = interval_bounds(forecast[0], lv.REPORTING, 0.90)
    hits = covered(lower, upper, actual)
    assert empirical_coverage(hits) >= 0.75  # fourteen days is a small sample


def test_history_too_short_for_the_rule_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="too short"):
        SeasonalNaive().forecast(np.zeros((1, 20)), horizon=14, levels=lv.REPORTING)


def test_a_misshaped_training_array_is_refused():
    with pytest.raises(ValueError, match="must be"):
        SeasonalNaive().forecast(np.zeros(100), horizon=7, levels=lv.REPORTING)
    with pytest.raises(ValueError, match="horizon must be"):
        SeasonalNaive().forecast(np.zeros((1, 100)), horizon=0, levels=lv.REPORTING)


def test_every_series_is_forecast_from_its_own_history():
    train = np.zeros((2, 200))
    train[0, :] = 5.0
    train[1, :] = 50.0
    forecast = SeasonalNaive().forecast(train, horizon=7, levels=lv.REPORTING)

    assert forecast[0].mean() == pytest.approx(5.0)
    assert forecast[1].mean() == pytest.approx(50.0)
    assert SEASON == 7


# --- the backtest runner ---------------------------------------------------------


def _small_panel(n_days: int = 900, seed: int = 0):
    """A two-leaf hierarchy with a weekly cycle, and a schedule that fits it."""
    from headroom.hierarchy.spec import from_paths

    hierarchy = from_paths([("BRONX", "B1"), ("BRONX", "B2")])
    rng = np.random.default_rng(seed)
    weekly = np.tile(np.array([80.0, 100.0, 100.0, 100.0, 100.0, 130.0, 90.0]), n_days // 7 + 1)
    leaves = np.stack(
        [
            weekly[:n_days] + rng.normal(0.0, 6.0, size=n_days),
            0.5 * weekly[:n_days] + rng.normal(0.0, 4.0, size=n_days),
        ]
    )
    days = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(n_days))
    schedule = Origins(days=days, min_train=400, horizon=14, step=7)
    return hierarchy, hierarchy.aggregate(leaves), schedule


def test_the_backtest_scores_every_origin_node_and_horizon_step():
    from headroom.backtest.run import run

    hierarchy, values, schedule = _small_panel()
    scores = run(SeasonalNaive(), "seasonal naive", values, hierarchy, schedule)

    assert scores.crps.shape == (len(schedule), hierarchy.n_nodes, schedule.horizon)
    assert scores.hits.shape[:3] == scores.crps.shape
    assert scores.model == "seasonal naive"
    assert np.all(np.isfinite(scores.crps))
    assert scores.fit_seconds > 0.0


def test_the_backtest_hands_the_model_only_what_the_origin_allows():
    """The runner's half of the look-ahead guarantee, checked by recording every call."""
    from headroom.backtest.run import run

    hierarchy, values, schedule = _small_panel()
    seen: list[int] = []

    class Recording:
        def forecast(self, train, horizon, levels):
            seen.append(train.shape[1])
            return SeasonalNaive().forecast(train, horizon, levels)

    run(Recording(), "recording", values, hierarchy, schedule)

    expected = [o.index + 1 for o in schedule]
    assert seen == expected
    # Every training array stops at its origin, so none of them reaches the last day.
    assert max(seen) <= len(schedule.days) - schedule.horizon


def test_a_sharper_model_scores_better_than_a_deliberately_hedged_one():
    """CRPS has to prefer the tighter of two models that both cover."""
    from headroom.backtest.run import run

    hierarchy, values, schedule = _small_panel()

    class Hedged:
        def forecast(self, train, horizon, levels):
            honest = SeasonalNaive().forecast(train, horizon, levels)
            median = honest[:, :, honest.shape[-1] // 2][:, :, np.newaxis]
            return median + 4.0 * (honest - median)

    honest = run(SeasonalNaive(), "honest", values, hierarchy, schedule)
    hedged = run(Hedged(), "hedged", values, hierarchy, schedule)

    assert honest.crps.mean() < hedged.crps.mean()
    assert hedged.widths.mean() > honest.widths.mean()


def test_by_origin_reduces_to_one_number_per_origin_for_the_bootstrap():
    from headroom.backtest.run import run

    hierarchy, values, schedule = _small_panel()
    scores = run(SeasonalNaive(), "seasonal naive", values, hierarchy, schedule)

    everything = scores.by_origin(scores.crps)
    city_only = scores.by_origin(scores.crps, "city")

    assert everything.shape == (len(schedule),)
    assert city_only.shape == (len(schedule),)
    # The city is a sum of two leaves, so its CRPS is on a larger scale than the average.
    assert city_only.mean() > everything.mean()


def test_a_panel_that_does_not_match_its_hierarchy_or_schedule_is_refused():
    from headroom.backtest.run import run

    hierarchy, values, schedule = _small_panel()

    with pytest.raises(ValueError, match="nodes"):
        run(SeasonalNaive(), "m", values[:-1], hierarchy, schedule)
    with pytest.raises(ValueError, match="days"):
        run(SeasonalNaive(), "m", values[:, :-1], hierarchy, schedule)


def test_the_two_quantile_grids_come_from_one_forecast():
    """The runner merges the grids so the tables and CRPS cannot disagree."""
    from headroom.backtest.run import _merge_grids

    scoring = np.array([0.1, 0.5, 0.9])
    reporting = np.array([0.05, 0.5, 0.95])
    combined, scoring_at, reporting_at = _merge_grids(scoring, reporting)

    assert combined.tolist() == [0.05, 0.1, 0.5, 0.9, 0.95]
    assert combined[scoring_at].tolist() == scoring.tolist()
    assert combined[reporting_at].tolist() == reporting.tolist()


# --- checkpointing and resume ----------------------------------------------------


def test_the_checkpointed_path_scores_identically_to_the_streaming_one(tmp_path):
    """Two drivers, one results table. If these diverge, a number depends on plumbing."""
    from headroom.backtest.driver import run_to_checkpoint, score_checkpoint
    from headroom.backtest.run import run
    from headroom.score import levels as lv

    hierarchy, values, schedule = _small_panel()

    streamed = run(SeasonalNaive(), "seasonal naive", values, hierarchy, schedule)
    checkpoint = run_to_checkpoint(
        SeasonalNaive(), "seasonal naive", values, hierarchy, schedule, tmp_path / "c.npz"
    )
    restored = score_checkpoint(
        checkpoint, "seasonal naive", values, hierarchy, schedule, lv.SCORING
    )

    np.testing.assert_allclose(streamed.crps, restored.crps)
    np.testing.assert_allclose(streamed.widths, restored.widths)
    np.testing.assert_array_equal(streamed.hits, restored.hits)
    np.testing.assert_allclose(streamed.reporting_quantiles, restored.reporting_quantiles)


def test_resuming_an_interrupted_run_produces_the_same_forecasts(tmp_path):
    from headroom.backtest.driver import run_to_checkpoint
    from headroom.backtest.store import open_checkpoint
    from headroom.score import levels as lv

    hierarchy, values, schedule = _small_panel()
    path = tmp_path / "c.npz"

    class Stops:
        """Forecasts a few origins, then refuses, as an interrupted run would."""

        def __init__(self, after: int) -> None:
            self.left = after

        def forecast(self, train, horizon, levels):
            if self.left <= 0:
                raise KeyboardInterrupt
            self.left -= 1
            return SeasonalNaive().forecast(train, horizon, levels)

    with pytest.raises(KeyboardInterrupt):
        run_to_checkpoint(
            Stops(12), "seasonal naive", values, hierarchy, schedule, path, save_every=5
        )

    partial = open_checkpoint(path, "seasonal naive", schedule, hierarchy.n_nodes, lv.SCORING)
    assert 0 < partial.n_done < len(schedule)
    assert not partial.complete

    resumed = run_to_checkpoint(
        SeasonalNaive(), "seasonal naive", values, hierarchy, schedule, path
    )
    assert resumed.complete

    whole = run_to_checkpoint(
        SeasonalNaive(), "seasonal naive", values, hierarchy, schedule, tmp_path / "w.npz"
    )
    np.testing.assert_allclose(resumed.quantiles, whole.quantiles)


def test_a_checkpoint_from_a_different_schedule_is_refused(tmp_path):
    from headroom.backtest.driver import run_to_checkpoint
    from headroom.backtest.store import open_checkpoint
    from headroom.score import levels as lv

    hierarchy, values, schedule = _small_panel()
    path = tmp_path / "c.npz"
    run_to_checkpoint(SeasonalNaive(), "seasonal naive", values, hierarchy, schedule, path)

    other = Origins(days=schedule.days, min_train=400, horizon=14, step=14)
    with pytest.raises(ValueError, match="different settings"):
        open_checkpoint(path, "seasonal naive", other, hierarchy.n_nodes, lv.SCORING)

    with pytest.raises(ValueError, match="different settings"):
        open_checkpoint(path, "a different model", schedule, hierarchy.n_nodes, lv.SCORING)


def test_scoring_a_partial_checkpoint_is_refused_rather_than_reported(tmp_path):
    from headroom.backtest.driver import score_checkpoint
    from headroom.backtest.store import open_checkpoint
    from headroom.score import levels as lv

    hierarchy, values, schedule = _small_panel()
    empty = open_checkpoint(
        tmp_path / "c.npz", "seasonal naive", schedule, hierarchy.n_nodes, lv.SCORING
    )
    with pytest.raises(ValueError, match="origins are done"):
        score_checkpoint(empty, "seasonal naive", values, hierarchy, schedule, lv.SCORING)


def test_score_forecasts_refuses_a_reporting_grid_outside_the_scoring_grid():
    from headroom.backtest.run import score_forecasts

    hierarchy, _, schedule = _small_panel()
    scoring = np.array([0.1, 0.5, 0.9])
    shape = (len(schedule), hierarchy.n_nodes, schedule.horizon)
    with pytest.raises(ValueError, match="subset of the scoring grid"):
        score_forecasts(
            "m",
            hierarchy,
            schedule,
            np.zeros((*shape, scoring.size)),
            np.zeros(shape),
            scoring,
            0.0,
            reporting=np.array([0.05, 0.5, 0.95]),
        )
