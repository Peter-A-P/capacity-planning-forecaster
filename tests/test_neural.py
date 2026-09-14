"""N-HiTS, fitted once and forecasting many times.

Skipped where NeuralForecast is not installed. CI does not install the neural extra: the
default Linux PyTorch wheel carries CUDA and is gigabytes, for tests that run on CPU.
Locally, `uv sync --extra neural` and these run in seconds on a deliberately tiny fit.
"""

import logging
import warnings

import numpy as np
import pytest

pytest.importorskip("neuralforecast")

from headroom.models.neural import SETTINGS, NHiTS

HORIZON = 7
LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


@pytest.fixture(autouse=True)
def _quiet():
    warnings.filterwarnings("ignore")
    for noisy in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _tiny(**overrides) -> NHiTS:
    settings = {
        **SETTINGS,
        "input_size": 28,
        "max_steps": 30,
        "windows_batch_size": 64,
        "batch_size": 3,
        **overrides,
    }
    return NHiTS(levels=LEVELS, horizon=HORIZON, settings=settings)


def _panel(n_days: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weekly = np.array([1.3, 1.1, 1.0, 0.95, 0.9, 0.8, 0.95])[np.arange(n_days) % 7]
    scale = np.array([4000.0, 800.0, 20.0])[:, np.newaxis]
    noise = rng.normal(1.0, 0.05, size=(3, n_days))
    return np.asarray(scale * weekly * noise, dtype=np.float64)


def test_forecasts_are_quantiles_on_the_requested_grid():
    model = _tiny()
    model.fit(_panel())
    q = model.predict(_panel())
    assert q.shape == (3, HORIZON, LEVELS.size)
    assert np.all(q >= 0.0)
    assert np.all(np.diff(q, axis=-1) >= 0.0)


def test_a_forecast_between_refits_reads_the_new_window_not_the_fitted_one():
    panel = _panel(n_days=260)
    model = _tiny()
    model.fit(panel[:, :200])
    at_fit = model.predict(panel[:, :200])
    later = model.predict(panel[:, :250])
    assert not np.array_equal(at_fit, later)
    # The same window and the same weights give the same forecast.
    np.testing.assert_array_equal(later, model.predict(panel[:, :250]))


def test_the_same_fit_twice_gives_the_same_forecast():
    first, second = _tiny(), _tiny()
    first.fit(_panel())
    second.fit(_panel())
    np.testing.assert_array_equal(first.predict(_panel()), second.predict(_panel()))


def test_predicting_before_fitting_is_refused():
    with pytest.raises(RuntimeError, match="fit before predicting"):
        _tiny().predict(_panel())


def test_a_window_shorter_than_one_example_is_refused():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_panel(n_days=30))


def test_the_protocol_call_refuses_a_different_grid_or_horizon():
    with pytest.raises(ValueError, match="one horizon and one quantile grid"):
        _tiny().forecast(_panel(), HORIZON + 1, LEVELS)
