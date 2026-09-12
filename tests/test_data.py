"""The loader's aggregation, the borough judgement call, and the calendar."""

from datetime import date

import polars as pl
import pytest

from headroom.data import calendar, checks
from headroom.data.nyc_ems import (
    _tidy,
    _year_windows,
    assign_areas_to_boroughs,
    daily_leaf_counts,
    fetch_daily_counts,
)


def _raw(rows: list[tuple[str, str, str, str]]) -> list[dict[str, str]]:
    return [{"day": d, "borough": b, "area": a, "n": n} for d, b, a, n in rows]


def test_tidy_normalises_staten_island_and_drops_rows_with_no_location():
    frame = _tidy(
        _raw(
            [
                ("2020-03-01T00:00:00.000", "RICHMOND / STATEN ISLAND", "S1", "28"),
                ("2020-03-01T00:00:00.000", "UNKNOWN", "B1", "3"),
                ("2020-03-01T00:00:00.000", "BRONX", "", "2"),
                ("2020-03-01T00:00:00.000", "bronx", " b1 ", "247"),
            ]
        )
    )
    assert frame.rows() == [
        (date(2020, 3, 1), "BRONX", "B1", 247),
        (date(2020, 3, 1), "STATEN ISLAND", "S1", 28),
    ]


def test_tidy_of_nothing_gives_an_empty_frame_with_the_right_schema():
    frame = _tidy([])
    assert frame.height == 0
    assert frame.schema["day"] == pl.Date
    assert frame.schema["n"] == pl.UInt32


def test_an_area_is_assigned_to_the_borough_it_is_recorded_under_most_often():
    frame = _tidy(
        _raw(
            [
                ("2020-03-01T00:00:00.000", "MANHATTAN", "M9", "1"),
                ("2020-03-01T00:00:00.000", "BRONX", "M9", "1"),
                ("2020-03-02T00:00:00.000", "MANHATTAN", "M9", "180"),
            ]
        )
    )
    assert assign_areas_to_boroughs(frame).rows() == [("M9", "MANHATTAN")]


def test_a_tie_is_broken_by_name_so_the_assignment_is_reproducible():
    frame = _tidy(
        _raw(
            [
                ("2020-03-01T00:00:00.000", "QUEENS", "X1", "5"),
                ("2020-03-01T00:00:00.000", "BRONX", "X1", "5"),
            ]
        )
    )
    assert assign_areas_to_boroughs(frame).rows() == [("X1", "BRONX")]


def test_leaf_counts_move_the_stray_incidents_to_the_assigned_borough():
    frame = _tidy(
        _raw(
            [
                ("2020-03-01T00:00:00.000", "MANHATTAN", "M9", "180"),
                ("2020-03-01T00:00:00.000", "BRONX", "M9", "1"),
            ]
        )
    )
    areas = assign_areas_to_boroughs(frame)
    leaf = daily_leaf_counts(frame, areas)

    assert leaf.rows() == [(date(2020, 3, 1), "MANHATTAN", "M9", 181)]
    # Nothing is lost in the move; the incidents are recounted under one borough.
    assert int(leaf["n"].sum()) == int(frame["n"].sum())


def test_the_disagreement_the_assignment_causes_is_reported_not_hidden():
    frame = _tidy(
        _raw(
            [
                ("2020-03-01T00:00:00.000", "MANHATTAN", "M9", "180"),
                ("2020-03-01T00:00:00.000", "BRONX", "M9", "1"),
                ("2020-03-01T00:00:00.000", "BRONX", "B1", "19"),
            ]
        )
    )
    report = checks.borough_disagreement(frame, assign_areas_to_boroughs(frame))

    assert report.total_incidents == 200
    assert report.moved_incidents == 1
    assert report.share == pytest.approx(0.005)
    assert report.worst_areas == (("M9", "MANHATTAN", 1),)


def test_a_day_the_dataset_is_missing_is_found_and_a_quiet_day_is_not():
    frame = _tidy(
        _raw(
            [
                ("2020-03-01T00:00:00.000", "BRONX", "B1", "10"),
                ("2020-03-03T00:00:00.000", "BRONX", "B1", "10"),
                ("2020-03-04T00:00:00.000", "BRONX", "B2", "1"),
            ]
        )
    )
    assert checks.missing_days(frame) == (date(2020, 3, 2),)


def test_missing_days_refuses_an_empty_frame_rather_than_reporting_none():
    with pytest.raises(ValueError, match="no rows"):
        checks.missing_days(_tidy([]))


def test_coverage_puts_the_thin_series_first():
    days = pl.date_range(date(2020, 1, 1), date(2020, 1, 10), interval="1d", eager=True)
    rows = [(d.isoformat() + "T00:00:00.000", "BRONX", "B1", "5") for d in days]
    rows += [(d.isoformat() + "T00:00:00.000", "BRONX", "B2", "5") for d in days[:2]]
    table = checks.coverage(_tidy(_raw(rows)))

    assert table["area"].to_list() == ["B2", "B1"]
    assert table["share"].to_list() == [0.2, 1.0]


def test_outliers_finds_a_one_day_spike():
    import numpy as np

    days = tuple(pl.date_range(date(2020, 1, 1), date(2021, 1, 1), interval="1d", eager=True))
    rng = np.random.default_rng(0)
    values = 1000.0 + rng.normal(0.0, 20.0, size=len(days))
    values[40] = 2000.0
    flagged = checks.outliers(values, days)

    assert [day for day, _, _ in flagged] == [days[40]]
    assert flagged[0][1] == 2000.0


def test_a_sustained_surge_is_flagged_where_a_standard_deviation_would_have_hidden_it():
    """The case this check exists for: March 2020 is a run of days, not one spike.

    Six weeks of elevated demand inflate a standard deviation enough to pull the surge
    back inside the threshold. The median absolute deviation is not moved by them.
    """
    import numpy as np

    days = tuple(pl.date_range(date(2020, 1, 1), date(2021, 1, 1), interval="1d", eager=True))
    rng = np.random.default_rng(0)
    values = 1000.0 + rng.normal(0.0, 20.0, size=len(days))
    surge = slice(60, 102)
    values[surge] += 300.0

    flagged = {day for day, _, _ in checks.outliers(values, days)}
    assert flagged == set(days[surge])

    z_by_sd = np.abs(values - values.mean()) / values.std()
    assert z_by_sd.max() < 6.0  # the same threshold finds nothing at all


def test_a_spike_on_an_otherwise_constant_series_is_still_found():
    """A zero median absolute deviation must not silently report no outliers."""
    import numpy as np

    days = tuple(pl.date_range(date(2020, 1, 1), date(2020, 3, 1), interval="1d", eager=True))
    values = np.full(len(days), 100.0)
    values[40] = 400.0
    assert [day for day, _, _ in checks.outliers(values, days)] == [days[40]]


def test_a_flat_series_has_no_outliers_rather_than_dividing_by_zero():
    import numpy as np

    days = tuple(pl.date_range(date(2020, 1, 1), date(2020, 1, 10), interval="1d", eager=True))
    assert checks.outliers(np.full(10, 7.0), days) == ()


def test_year_windows_cover_a_part_year_range_exactly_once():
    windows = list(_year_windows(date(2019, 6, 1), date(2021, 3, 1)))
    assert windows == [
        (2019, date(2019, 6, 1), date(2020, 1, 1)),
        (2020, date(2020, 1, 1), date(2021, 1, 1)),
        (2021, date(2021, 1, 1), date(2021, 3, 1)),
    ]


def test_fetch_refuses_a_window_the_dataset_does_not_cover(tmp_path):
    with pytest.raises(ValueError, match="dataset starts"):
        fetch_daily_counts(date(2004, 1, 1), date(2005, 1, 1), tmp_path)
    with pytest.raises(ValueError, match="empty window"):
        fetch_daily_counts(date(2020, 1, 1), date(2020, 1, 1), tmp_path)


def test_the_federal_holidays_land_on_dates_anyone_can_check():
    h2020 = calendar.federal_holidays(2020)
    assert h2020[date(2020, 1, 20)] == "mlk"
    assert h2020[date(2020, 5, 25)] == "memorial"
    assert h2020[date(2020, 9, 7)] == "labor"
    assert h2020[date(2020, 11, 26)] == "thanksgiving"
    assert h2020[date(2020, 7, 4)] == "independence"

    h2024 = calendar.federal_holidays(2024)
    assert h2024[date(2024, 1, 15)] == "mlk"
    assert h2024[date(2024, 5, 27)] == "memorial"
    assert h2024[date(2024, 11, 28)] == "thanksgiving"


def test_juneteenth_starts_in_2021_rather_than_backfilling_twenty_years():
    assert date(2020, 6, 19) not in calendar.federal_holidays(2020)
    assert calendar.federal_holidays(2021)[date(2021, 6, 19)] == "juneteenth"


def test_a_weekend_holiday_is_observed_on_the_nearest_weekday():
    assert calendar.observed(date(2021, 7, 4)) == date(2021, 7, 5)  # Sunday
    assert calendar.observed(date(2020, 7, 4)) == date(2020, 7, 3)  # Saturday
    assert calendar.observed(date(2019, 7, 4)) == date(2019, 7, 4)  # Thursday


def test_features_date_demand_by_the_day_the_holiday_falls_on():
    days = tuple(pl.date_range(date(2021, 7, 3), date(2021, 7, 6), interval="1d", eager=True))
    frame = calendar.features(days)

    assert frame["dow"].to_list() == [5, 6, 0, 1]
    assert frame["is_weekend"].to_list() == [True, True, False, False]
    assert frame["holiday"].to_list() == [None, "independence", None, None]
    assert frame["is_holiday"].to_list() == [False, True, False, False]
    assert frame["is_holiday_observed"].to_list() == [False, False, True, False]


def test_features_refuses_an_empty_window():
    with pytest.raises(ValueError, match="no days"):
        calendar.features(())
