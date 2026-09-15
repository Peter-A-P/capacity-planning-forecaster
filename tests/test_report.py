"""The results tables and the one place the README is written.

The README's tables are only ever written by `headroom report`, between two markers. The
tests that matter: nothing outside the markers changes, a README without them is refused
rather than guessed at, running the report twice gives the same file, and an interval
never loses its sign or its precision in the formatting.
"""

from pathlib import Path

import numpy as np
import pytest

from headroom.cli import _method_label
from headroom.report.tables import (
    END,
    START,
    Estimate,
    decimals_for,
    duration,
    markdown_table,
    number,
    splice,
    worst_window,
)

README = (
    f"# Title\n\nIntro stays.\n\n## Result\n\n{START}\nold tables\n{END}\n\n## After\n\nKept.\n"
)


def test_splice_replaces_only_what_lies_between_the_markers():
    out = splice(README, "new tables")
    assert out == README.replace("old tables", "new tables")


def test_splice_is_idempotent():
    once = splice(README, "| a |\n|---|\n| 1 |\n")
    assert splice(once, "| a |\n|---|\n| 1 |\n") == once


@pytest.mark.parametrize(
    "text",
    [
        "# No markers at all\n",
        f"{START}\nonly the start\n",
        f"{START}\n{START}\ntwice\n{END}\n",
        f"{END}\nend first\n{START}\n",
    ],
)
def test_splice_refuses_a_readme_whose_markers_are_not_one_ordered_pair(text):
    with pytest.raises(ValueError, match="marker"):
        splice(text, "tables")


def test_the_real_readme_has_its_markers():
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert text.count(START) == 1
    assert text.count(END) == 1
    assert text.index(START) < text.index(END)


def test_a_signed_interval_keeps_its_signs_and_places():
    assert number(Estimate(-0.0165, -0.068, 0.0598), places=3, signed=True) == (
        "-0.017 [-0.068, +0.060]"
    )


@pytest.mark.parametrize(
    ("value", "expected"), [(135.4, 0), (32.39, 1), (8.27, 2), (0.904, 3), (-28.4, 1)]
)
def test_decimals_give_about_three_significant_figures(value, expected):
    assert decimals_for(value) == expected


def test_a_tie_never_reads_as_negative_zero():
    assert number(Estimate(-0.001, -0.57, 0.69), places=2, signed=True) == (
        "+0.00 [-0.57, +0.69]"
    )


def test_number_chooses_places_from_the_point_estimate():
    assert number(Estimate(135.4, 128.2, 144.9)) == "135 [128, 145]"


@pytest.mark.parametrize(
    ("seconds", "expected"), [(5.0, "1 min"), (240.0, "4 min"), (65_880.0, "18.3 h")]
)
def test_duration(seconds, expected):
    assert duration(seconds) == expected


def test_duration_refuses_a_negative_time():
    with pytest.raises(ValueError, match="non-negative"):
        duration(-1.0)


def test_markdown_table_rejects_a_short_row_and_a_pipe():
    markdown_table(["a", "b"], [["1", "2"]])
    with pytest.raises(ValueError, match="cells"):
        markdown_table(["a", "b"], [["1"]])
    with pytest.raises(ValueError, match="pipe"):
        markdown_table(["a"], [["x|y"]])


def test_worst_window_is_trailing_and_dated_by_its_last_origin():
    coverage = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    at, value = worst_window(coverage, window=2)
    assert (at, value) == (4, 0.0)


def test_worst_window_ignores_positions_before_the_window_is_full():
    # A short-window average at the start would report 0.0 at position 0.
    coverage = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    at, value = worst_window(coverage, window=3)
    assert (at, value) == (2, pytest.approx(2 / 3))


def test_worst_window_refuses_a_window_longer_than_the_series():
    with pytest.raises(ValueError, match="does not fit"):
        worst_window(np.ones(3), window=4)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("LightGBM", "LightGBM, global"),
        ("N-HiTS-refit4", "N-HiTS, refitted every 4 weeks"),
        ("ETS + MinT", "ETS + MinT"),
        ("Seasonal naive", "Seasonal naive"),
    ],
)
def test_method_labels(name, label):
    assert _method_label(name) == label
