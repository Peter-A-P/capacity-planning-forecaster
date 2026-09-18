"""The results tables and the one place the README is written.

The README's tables are only ever written by `headroom report`, between two markers. The
tests that matter: nothing outside the markers changes, a README without them is refused
rather than guessed at, running the report twice gives the same file, and an interval
never loses its sign or its precision in the formatting.
"""

from pathlib import Path

import numpy as np
import pytest

from headroom.cli import (
    MIN_BOOTSTRAP_ORIGINS,
    PROMISED_ROWS,
    _absent_rows,
    _base_model,
    _difference_cell,
    _listing,
    _method_label,
    _refit_every,
)
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
from headroom.score.bootstrap import BLOCK

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
    assert at == 2
    assert value == pytest.approx(2 / 3)


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


@pytest.mark.parametrize(
    ("name", "network", "every"),
    [
        ("PatchTST-refit13", "PatchTST", 13),
        ("N-HiTS-refit4", "N-HiTS", 4),
        ("LightGBM", "LightGBM", None),
        ("ETS", "ETS", None),
    ],
)
def test_a_checkpoint_name_splits_into_network_and_schedule(name, network, every):
    assert _base_model(name) == network
    assert _refit_every(name) == every


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], ""),
        (["a"], "a"),
        (["a", "b"], "a and b"),
        (["a", "b", "c"], "a, b and c"),
    ],
)
def test_listing_reads_as_a_sentence(items, expected):
    assert _listing(items) == expected


def test_a_network_that_was_run_is_never_also_listed_as_not_built():
    # The bug this guards: a PatchTST checkpoint was reported and the table still carried
    # a hardcoded "PatchTST | not built" row beneath it, saying both things at once.
    reported = ["LightGBM", "N-HiTS-refit4", "PatchTST-refit13"]
    built = {_base_model(name) for name in reported}
    assert _absent_rows(built, set()) == [("TimesFM, zero-shot", "not built")]


def test_a_model_scored_on_its_own_windows_is_not_called_not_built_either():
    # TimesFM is kept out of the main table because its origins are not that table's
    # origins. That is a different statement from not existing, and a reader who is told
    # "not built" under a table of TimesFM results has been told something false.
    built = {"LightGBM", "N-HiTS", "PatchTST"}
    assert _absent_rows(built, {"TimesFM"}) == [("TimesFM, zero-shot", "own windows, below")]


def test_a_model_in_the_main_table_gets_no_absent_row_whatever_else_it_is_in():
    assert _absent_rows({"N-HiTS", "PatchTST", "TimesFM"}, {"TimesFM"}) == []


def test_a_window_too_short_for_the_block_bootstrap_reports_no_interval():
    # Measured on TimesFM's 40-origin cross-check window: the bootstrap returned an
    # interval that did not contain its own point estimate. A narrow wrong interval is
    # worse than none, because it reads as precision.
    rng = np.random.default_rng(0)
    short = rng.normal(size=MIN_BOOTSTRAP_ORIGINS - 1)
    cell = _difference_cell(short, places=2)
    assert "[" not in cell
    assert cell.startswith(("+", "-"))


def test_a_window_long_enough_still_carries_its_interval():
    rng = np.random.default_rng(0)
    values = rng.normal(size=MIN_BOOTSTRAP_ORIGINS)
    assert "[" in _difference_cell(values, places=2)


def test_the_floor_is_whole_blocks_of_the_bootstrap_this_project_uses():
    # If BLOCK changes, the floor has to move with it rather than stay at an old number.
    assert MIN_BOOTSTRAP_ORIGINS == 4 * BLOCK


def test_every_promised_network_is_named_the_way_its_checkpoints_are():
    # PROMISED_ROWS is matched against _base_model output, so its keys have to be the
    # network names, not table labels, or a built model would never match its promise.
    for network, _ in PROMISED_ROWS:
        assert _base_model(f"{network}-refit7") == network
