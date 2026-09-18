"""The two charts `PLAN.md` section 1 promises, written to disk as PNG.

Both exist to show something a table cannot. The coverage chart shows **when** an interval
stopped working, which is the whole March 2020 finding; a whole-period coverage number
averages that away into a figure near nominal. The fan chart shows what the failure looked
like: the band, the outcome, and the distance between them, in incidents a planner counts.

## Coverage is never plotted without width

A conformal method reaches nominal coverage trivially by being wide enough to cover
anything, so a coverage line on its own can be read as success when it is only caution.
Every coverage panel here has the mean width of the same intervals under it, on the same
time axis, and :func:`coverage_chart` will not draw one without the other.

## No interactive backend

Matplotlib's Agg backend is selected before pyplot is imported, so these run headless in CI
and on a machine with no display. Nothing here opens a window.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

#: Colours for the methods, in the order they are given. Chosen to stay distinguishable in
#: greyscale and to the most common colour-vision deficiencies: they differ in lightness as
#: well as in hue, and no red is paired with a green.
PALETTE: Final[tuple[str, ...]] = (
    "#1b1b1b",
    "#1f77b4",
    "#d95f02",
    "#7570b3",
    "#117733",
)

#: The band edges a fan chart draws, widest first, as (lower level, upper level).
FAN_BANDS: Final[tuple[tuple[float, float], ...]] = (
    (0.025, 0.975),
    (0.05, 0.95),
    (0.10, 0.90),
    (0.25, 0.75),
)

#: Figure width in inches. Tall enough per panel to read a coverage line against nominal.
WIDTH_INCHES: Final[float] = 11.0
PANEL_INCHES: Final[float] = 2.4


def _pyplot() -> Any:  # noqa: ANN401 - matplotlib is untyped here
    """Import pyplot with a non-interactive backend already chosen.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True, slots=True)
class CoverageSeries:
    """One method's coverage and width through time, at one nominal level.

    Attributes:
        method: The method's name, as the legend shows it.
        nominal: The nominal coverage these intervals claim.
        coverage: Coverage at each origin, ``nan`` before the method could calibrate.
        width: Mean interval width at each origin, ``nan`` in the same places.
    """

    method: str
    nominal: float
    coverage: npt.NDArray[np.float64]
    width: npt.NDArray[np.float64]


def rolling_mean(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Trailing mean over ``window`` points, ignoring the gaps before calibration starts.

    Args:
        values: One value per origin, possibly with ``nan`` at the front.
        window: Points in the trailing window.

    Returns:
        The trailing mean, ``nan`` until the window is full of finite values.

    Raises:
        ValueError: The window does not fit the series.
    """
    if window < 1 or window > values.size:
        raise ValueError(f"a {window}-point window does not fit {values.size} points")
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones(window)
    totals = np.convolve(filled, kernel, mode="valid")
    counts = np.convolve(finite.astype(np.float64), kernel, mode="valid")
    out = np.full(values.size, np.nan)
    with np.errstate(invalid="ignore"):
        out[window - 1 :] = np.where(counts == window, totals / window, np.nan)
    return out


def coverage_chart(
    series: list[CoverageSeries],
    origin_days: list[date],
    path: Path,
    window: int,
    level_name: str,
    shift: tuple[date, date] | None = None,
) -> Path:
    """Draw rolling coverage against nominal, with the widths that produced it.

    One row per nominal level, each with coverage above and mean width below, so a method
    that reaches nominal by widening is visible as such rather than as a success.

    Args:
        series: Every method at every nominal level. Grouped into rows by nominal.
        origin_days: The day of each origin, the same length as each series.
        path: Where to write the PNG. Its parent is created.
        window: Origins in the trailing window.
        level_name: The hierarchy level these are measured at, for the title.
        shift: A period to shade and label, such as the 2020 shift.

    Returns:
        ``path``.

    Raises:
        ValueError: No series, or a series whose length does not match ``origin_days``.
    """
    if not series:
        raise ValueError("nothing to plot")
    for one in series:
        if one.coverage.size != len(origin_days) or one.width.size != len(origin_days):
            raise ValueError(
                f"{one.method} has {one.coverage.size} origins, {len(origin_days)} days given"
            )

    plt = _pyplot()
    nominals = sorted({one.nominal for one in series}, reverse=True)
    figure, axes = plt.subplots(
        2 * len(nominals),
        1,
        figsize=(WIDTH_INCHES, PANEL_INCHES * 2 * len(nominals)),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * len(nominals), "hspace": 0.12},
    )
    axes = np.atleast_1d(axes)
    days = np.array(origin_days)

    for row, nominal in enumerate(nominals):
        top, bottom = axes[2 * row], axes[2 * row + 1]
        here = [one for one in series if one.nominal == nominal]
        for i, one in enumerate(here):
            colour = PALETTE[i % len(PALETTE)]
            top.plot(
                days,
                rolling_mean(one.coverage, window),
                color=colour,
                linewidth=1.3,
                label=one.method,
            )
            bottom.plot(days, rolling_mean(one.width, window), color=colour, linewidth=1.1)
        top.axhline(nominal, color="#888888", linestyle="--", linewidth=1.0)
        top.set_ylabel(f"coverage at {nominal:.0%}")
        bottom.set_ylabel("mean width")
        for axis in (top, bottom):
            axis.grid(True, alpha=0.25, linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            if shift is not None:
                axis.axvspan(shift[0], shift[1], color="#d95f02", alpha=0.10, linewidth=0)
        if row == 0:
            top.legend(loc="lower left", frameon=False, fontsize=8, ncol=len(here))
            if shift is not None:
                # At the top of the panel, where the legend is not: the lines sink during
                # the shift, so the bottom-left corner is exactly where the story is.
                top.annotate(
                    "demand shift",
                    xy=(shift[1], top.get_ylim()[1]),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=8,
                    color="#d95f02",
                )

    axes[0].set_title(
        f"Coverage through time at the {level_name}, trailing {window} origins, "
        "with the widths that produced it",
        loc="left",
        fontsize=11,
    )
    axes[-1].set_xlabel("forecast origin")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def fan_chart(
    quantiles: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    days: list[date],
    names: list[str],
    path: Path,
    horizon_step: int,
    shift: tuple[date, date] | None = None,
) -> Path:
    """Draw forecast bands against what happened, one panel per series.

    Args:
        quantiles: Forecasts, shape ``(n_days, n_series, n_levels)``, one row per target
            day at a fixed horizon step.
        levels: The quantile grid ``quantiles`` is on.
        actual: What happened, shape ``(n_days, n_series)``.
        days: The target day of each row.
        names: One label per series, as the panel titles.
        path: Where to write the PNG. Its parent is created.
        horizon_step: Days ahead these forecasts were made, for the title.
        shift: A period to shade, such as the 2020 shift.

    Returns:
        ``path``.

    Raises:
        ValueError: The shapes disagree with each other or with ``days`` and ``names``.
    """
    if quantiles.ndim != 3 or quantiles.shape[:2] != actual.shape:
        raise ValueError(f"quantiles {quantiles.shape} and actual {actual.shape} disagree")
    if quantiles.shape[0] != len(days) or quantiles.shape[1] != len(names):
        raise ValueError(f"{len(days)} days and {len(names)} names for {quantiles.shape}")
    if quantiles.shape[2] != levels.size:
        raise ValueError(f"{levels.size} levels for {quantiles.shape[2]} columns")

    plt = _pyplot()
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(WIDTH_INCHES, PANEL_INCHES * 1.5 * len(names)),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    axes = np.atleast_1d(axes)
    when = np.array(days)

    for panel, (axis, name) in enumerate(zip(axes, names, strict=True)):
        for shade, (low, high) in enumerate(FAN_BANDS):
            lower = quantiles[:, panel, int(np.abs(levels - low).argmin())]
            upper = quantiles[:, panel, int(np.abs(levels - high).argmin())]
            axis.fill_between(
                when,
                lower,
                upper,
                color="#1f77b4",
                alpha=0.13 + 0.09 * shade,
                linewidth=0,
                label=f"{high - low:.0%}" if panel == 0 else None,
            )
        axis.plot(
            when,
            actual[:, panel],
            color="#1b1b1b",
            linewidth=1.0,
            label="actual" if panel == 0 else None,
        )
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        if shift is not None:
            axis.axvspan(shift[0], shift[1], color="#d95f02", alpha=0.10, linewidth=0)
        if panel == 0:
            axis.legend(loc="upper left", frameon=False, fontsize=8, ncol=5)

    axes[0].set_title(
        f"Forecast bands {horizon_step} days ahead, against what happened",
        loc="left",
        fontsize=11,
    )
    axes[-1].set_xlabel("day being forecast")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
