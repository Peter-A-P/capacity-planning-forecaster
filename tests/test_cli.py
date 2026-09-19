"""Where the command line reads and writes backtest output.

A weekly statistical backtest is eight hours of CPU. The test that matters here is that
the command which scores checkpoints looks for exactly the file names the command which
wrote them used, including the names of checkpoints that already exist on disk.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from headroom.backtest.origins import Origins
from headroom.cli import (
    DEFAULT_OUT,
    OUT_VARIABLE,
    _checkpoint_name,
    _print_pinball,
    output_dir,
)
from headroom.score import levels as lv

if TYPE_CHECKING:  # The return type names a module the tests do not otherwise import.
    from headroom.backtest.run import Scores

DAYS = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(2_000))


def _scored() -> "Scores":
    """Seasonal naive over a small two-leaf hierarchy, scored."""
    from headroom.backtest.run import run
    from headroom.hierarchy.spec import from_paths
    from headroom.models.baselines import SeasonalNaive

    hierarchy = from_paths([("BRONX", "B1"), ("BRONX", "B2")])
    rng = np.random.default_rng(0)
    n_days = 900
    weekly = np.tile(np.array([80.0, 100.0, 100.0, 100.0, 100.0, 130.0, 90.0]), n_days // 7 + 1)
    leaves = np.stack(
        [
            weekly[:n_days] + rng.normal(0.0, 6.0, size=n_days),
            0.5 * weekly[:n_days] + rng.normal(0.0, 4.0, size=n_days),
        ]
    )
    days = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(n_days))
    schedule = Origins(days=days, min_train=400, horizon=14, step=7, train_window=400)
    return run(SeasonalNaive(), "naive", hierarchy.aggregate(leaves), hierarchy, schedule)


def test_pinball_is_printed_for_every_reporting_quantile_and_level(capsys):
    scores = _scored()
    _print_pinball(scores, scores)
    table = [line for line in capsys.readouterr().out.splitlines() if line.startswith("|")]

    # Header, separator, then one row per reporting quantile.
    assert len(table) == 2 + len(lv.REPORTING)
    assert table[0].count("|") == 5  # quantile, then the three hierarchy levels
    for line, level in zip(table[2:], lv.REPORTING, strict=True):
        assert line.startswith(f"| {level:.3f} |")


def test_a_model_scored_against_itself_has_no_skill_at_any_quantile(capsys):
    # The axis test. Pinball is indexed by quantile last, and reading the wrong axis here
    # would still print a plausible-looking table of numbers.
    scores = _scored()
    _print_pinball(scores, scores)
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("| 0."):
            assert line.count("+0.0000 [+0.0000, +0.0000]") == 3, line


def test_output_defaults_to_backtest_out(monkeypatch):
    monkeypatch.delenv(OUT_VARIABLE, raising=False)
    assert output_dir() == DEFAULT_OUT == Path("backtest/out")


def test_output_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(OUT_VARIABLE, str(tmp_path / "elsewhere"))
    assert output_dir() == tmp_path / "elsewhere"


def test_a_blank_variable_is_the_default_not_the_current_directory(monkeypatch):
    # Path("") is ".", which would scatter 740 MB checkpoints into wherever the command
    # happened to be run from.
    monkeypatch.setenv(OUT_VARIABLE, "   ")
    assert output_dir() == DEFAULT_OUT


def test_the_variable_is_read_when_called_not_when_imported(monkeypatch, tmp_path):
    monkeypatch.delenv(OUT_VARIABLE, raising=False)
    before = output_dir()
    monkeypatch.setenv(OUT_VARIABLE, str(tmp_path))
    assert (before, output_dir()) == (DEFAULT_OUT, tmp_path)


@pytest.mark.parametrize(
    ("train_window", "expected"),
    [(1_095, "ETS-step7-win1095.npz"), (None, "ETS-step7-win0.npz")],
)
def test_checkpoint_names_match_the_files_the_weekly_run_wrote(train_window, expected):
    origins = Origins(days=DAYS, min_train=1_095, horizon=14, step=7, train_window=train_window)
    assert _checkpoint_name("ETS", 7, origins) == expected
