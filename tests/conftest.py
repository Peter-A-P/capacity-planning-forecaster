"""Opt-in flags for the tests that cost real time or reach the network.

`pyproject.toml` declares the markers; this is what makes them mean something. Without it
a marker is only a label and the expensive tests run on every push anyway.
"""

import pytest

_OPTIONAL = {
    "slow": ("--run-slow", "a full-size fit or a whole backtest"),
    "network": ("--run-network", "fetches from NYC Open Data"),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in flags.

    Args:
        parser: The pytest argument parser.
    """
    for marker, (flag, help_text) in _OPTIONAL.items():
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run tests marked {marker}: {help_text}",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the opt-in tests unless their flag was given.

    Args:
        config: The pytest config.
        items: The collected tests.
    """
    for marker, (flag, help_text) in _OPTIONAL.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag} ({help_text})")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
