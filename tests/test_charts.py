"""The charts: the rolling window, the refusals, and that a file actually appears.

A chart cannot be asserted to be readable, so these test the parts that can be wrong
silently: the trailing mean that decides what every line shows, the refusals that stop a
mismatched series being drawn against the wrong dates, and that the writer produces a real
PNG rather than an empty file.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from headroom.report.charts import (
    FAN_BANDS,
    PALETTE,
    CoverageSeries,
    coverage_chart,
    fan_chart,
    rolling_mean,
)

DAYS = [date(2020, 1, 5) + timedelta(days=7 * i) for i in range(40)]
PNG = b"\x89PNG\r\n\x1a\n"


def test_the_rolling_mean_is_trailing_and_starts_when_the_window_is_full():
    values = np.arange(6, dtype=float)
    out = rolling_mean(values, 3)
    assert np.isnan(out[:2]).all()
    # The value at position 2 averages positions 0, 1 and 2, not 2, 3 and 4.
    assert out[2] == pytest.approx(1.0)
    assert out[5] == pytest.approx(4.0)


def test_the_rolling_mean_waits_for_a_window_with_no_gaps_in_it():
    # Coverage is nan before a method has calibrated. Averaging over a window that still
    # contains one of those would put a number on the chart earlier than the method could
    # produce one, which is exactly the kind of quiet overclaim these charts exist to avoid.
    values = np.array([np.nan, 1.0, 1.0, 1.0, 1.0])
    out = rolling_mean(values, 3)
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx(1.0)


def test_the_rolling_window_must_fit():
    with pytest.raises(ValueError, match="does not fit"):
        rolling_mean(np.ones(3), 4)


def _series(nominal: float = 0.9, method: str = "split") -> CoverageSeries:
    rng = np.random.default_rng(0)
    return CoverageSeries(
        method=method,
        nominal=nominal,
        coverage=rng.uniform(0.7, 1.0, size=len(DAYS)),
        width=rng.uniform(500.0, 900.0, size=len(DAYS)),
    )


def test_the_coverage_chart_writes_a_png(tmp_path):
    path = coverage_chart(
        [_series(0.9, "split"), _series(0.9, "adaptive"), _series(0.8, "split")],
        DAYS,
        tmp_path / "nested" / "coverage.png",
        window=4,
        level_name="city",
        shift=(date(2020, 3, 1), date(2020, 6, 1)),
    )
    assert path.exists()
    assert path.read_bytes()[:8] == PNG


def test_the_coverage_chart_refuses_a_series_of_the_wrong_length(tmp_path):
    # Silently drawing 39 values against 40 dates would shift every line by a week and
    # nothing about the picture would look wrong.
    wrong = CoverageSeries("split", 0.9, np.ones(len(DAYS) - 1), np.ones(len(DAYS) - 1))
    with pytest.raises(ValueError, match="origins"):
        coverage_chart([wrong], DAYS, tmp_path / "c.png", window=4, level_name="city")


def test_the_coverage_chart_refuses_to_draw_nothing(tmp_path):
    with pytest.raises(ValueError, match="nothing to plot"):
        coverage_chart([], DAYS, tmp_path / "c.png", window=4, level_name="city")


def _fan(n_days: int = 30, n_series: int = 2, n_levels: int = 199):
    levels = np.linspace(0.005, 0.995, n_levels)
    centre = np.linspace(100.0, 200.0, n_days)[:, None, None]
    spread = (levels - 0.5)[None, None, :] * 80.0
    quantiles = centre + spread + np.arange(n_series)[None, :, None] * 10.0
    actual = quantiles[:, :, n_levels // 2]
    days = [date(2020, 2, 1) + timedelta(days=i) for i in range(n_days)]
    names = [f"series {i}" for i in range(n_series)]
    return quantiles, levels, actual, days, names


def test_the_fan_chart_writes_a_png(tmp_path):
    quantiles, levels, actual, days, names = _fan()
    path = fan_chart(
        quantiles, levels, actual, days, names, tmp_path / "fan.png", horizon_step=14
    )
    assert path.exists()
    assert path.read_bytes()[:8] == PNG


def test_the_fan_chart_refuses_shapes_that_do_not_line_up(tmp_path):
    quantiles, levels, actual, days, names = _fan()
    with pytest.raises(ValueError, match="disagree"):
        fan_chart(
            quantiles, levels, actual[:-1], days, names, tmp_path / "f.png", horizon_step=14
        )
    with pytest.raises(ValueError, match="names"):
        fan_chart(
            quantiles, levels, actual, days, ["only one"], tmp_path / "f.png", horizon_step=14
        )
    with pytest.raises(ValueError, match="levels"):
        fan_chart(
            quantiles, levels[:-1], actual, days, names, tmp_path / "f.png", horizon_step=14
        )


def test_the_fan_bands_are_central_and_widest_first():
    # The shading darkens with each band, so a band out of order would paint a wide band
    # over a narrow one and the chart would read as the wrong interval.
    for low, high in FAN_BANDS:
        assert low + high == pytest.approx(1.0)
    spans = [high - low for low, high in FAN_BANDS]
    assert spans == sorted(spans, reverse=True)


def test_there_are_enough_colours_for_the_methods_drawn_together():
    assert len(PALETTE) >= 3
    assert len(set(PALETTE)) == len(PALETTE)
