"""Where the command line reads and writes backtest output.

A weekly statistical backtest is eight hours of CPU. The test that matters here is that
the command which scores checkpoints looks for exactly the file names the command which
wrote them used, including the names of checkpoints that already exist on disk.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from headroom.backtest.origins import Origins
from headroom.cli import DEFAULT_OUT, OUT_VARIABLE, _checkpoint_name, output_dir

DAYS = tuple(date(2010, 1, 1) + timedelta(days=i) for i in range(2_000))


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
