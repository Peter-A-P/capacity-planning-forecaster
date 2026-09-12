"""The StatsForecast wrapper.

The pure parts, which are where the bugs would be, run without fitting anything. The one
real fit is marked `slow`: it takes seconds, not the minutes a full-panel fit takes, and
it is the only test that proves the column mapping matches what StatsForecast actually
emits rather than what its documentation says.
"""

import numpy as np
import pandas as pd
import pytest

from headroom.models.stats import (
    EPOCH,
    StatisticalModel,
    _long_frame,
    catalogue,
    column_for,
    required_levels,
)
from headroom.score import levels as lv


def test_the_reporting_grid_maps_onto_five_symmetric_intervals():
    assert required_levels(lv.REPORTING) == [50, 80, 90, 95, 98]


def test_the_scoring_grid_maps_onto_the_ninety_nine_integer_levels():
    assert required_levels(lv.SCORING) == list(range(1, 100))


def test_the_median_needs_no_interval_and_reads_the_point_forecast():
    assert required_levels(np.array([0.5])) == []
    assert column_for(0.5, "AutoETS") == "AutoETS"


def test_a_quantile_below_the_median_reads_the_low_bound_and_above_it_the_high():
    assert column_for(0.05, "AutoETS") == "AutoETS-lo-90"
    assert column_for(0.95, "AutoETS") == "AutoETS-hi-90"
    assert column_for(0.01, "MSTL") == "MSTL-lo-98"
    assert column_for(0.75, "AutoARIMA") == "AutoARIMA-hi-50"


def test_a_quantile_needing_a_fractional_level_is_refused_rather_than_rounded():
    """0.123 needs level 75.4, which StatsForecast cannot be asked for."""
    with pytest.raises(ValueError, match=r"not an\s+integer"):
        required_levels(np.array([0.123, 0.5, 0.877]))


def test_the_long_frame_keeps_every_series_contiguous_and_in_order():
    train = np.arange(12.0).reshape(3, 4)
    frame = _long_frame(train)

    assert list(frame.columns) == ["unique_id", "ds", "y"]
    assert frame.shape == (12, 3)
    assert frame["unique_id"].tolist()[:4] == ["s0000"] * 4
    assert frame["y"].tolist()[:4] == [0.0, 1.0, 2.0, 3.0]
    assert frame["y"].tolist()[-4:] == [8.0, 9.0, 10.0, 11.0]
    # String ordering has to match series ordering, because the forecast is reshaped back.
    assert sorted(frame["unique_id"].unique()) == ["s0000", "s0001", "s0002"]


def test_the_frame_carries_synthetic_dates_so_no_model_can_read_the_calendar():
    frame = _long_frame(np.zeros((2, 5)))
    assert frame["ds"].min() == pd.Timestamp(EPOCH)
    gaps = frame["ds"].to_numpy().astype("datetime64[D]")
    within_first_series = np.diff(gaps[:5]).astype(int)
    assert within_first_series.tolist() == [1, 1, 1, 1]


def test_ten_series_of_padding_cannot_shift_a_series_into_another_ones_rows():
    """Regression guard on the reshape: series 2's values must stay series 2's."""
    train = np.vstack([np.full(6, float(i)) for i in range(10)])
    frame = _long_frame(train)
    for i in range(10):
        rows = frame[frame["unique_id"] == f"s{i:04d}"]
        assert rows["y"].tolist() == [float(i)] * 6


def test_the_catalogue_is_the_four_models_the_plan_names():
    names = [m.name for m in catalogue()]
    assert names == ["ETS", "Theta", "MSTL", "AutoARIMA"]
    assert all(callable(m.build) for m in catalogue())


def test_each_catalogue_entry_builds_a_fresh_model_rather_than_sharing_one():
    """A fitted model carried from one origin to the next would be look-ahead."""
    model = catalogue()[0]
    assert model.build() is not model.build()


def test_a_misshaped_training_array_is_refused():
    model = catalogue()[0]
    with pytest.raises(ValueError, match="train must be"):
        model.forecast(np.zeros(50), 14, lv.REPORTING)
    with pytest.raises(ValueError, match="horizon must be"):
        model.forecast(np.zeros((2, 50)), 0, lv.REPORTING)


@pytest.mark.slow
def test_a_real_fit_produces_sorted_non_negative_quantiles_in_the_right_shape():
    """The only test that proves the column mapping against StatsForecast itself."""
    rng = np.random.default_rng(0)
    weekly = np.tile(np.array([80.0, 100.0, 100.0, 100.0, 100.0, 130.0, 90.0]), 40)
    train = np.vstack([weekly + rng.normal(0.0, 5.0, size=weekly.size) for _ in range(2)])

    model = StatisticalModel("ETS", catalogue()[0].build, "AutoETS", n_jobs=1)
    out = model.forecast(train, horizon=14, levels=lv.REPORTING)

    assert out.shape == (2, 14, lv.REPORTING.size)
    assert np.all(np.diff(out, axis=-1) >= 0.0)
    assert np.all(out >= 0.0)
    # The median should be in the neighbourhood of the weekly pattern it was shown.
    median = out[0, :, lv.REPORTING.tolist().index(0.5)]
    assert 60.0 < float(median.mean()) < 150.0
