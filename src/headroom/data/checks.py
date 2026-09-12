"""Checks run over the loaded counts before anything is modelled.

None of these fixes anything. Each one measures something that would otherwise be an
assumption, and `docs/data.md` prints the results, so a reader can see the size of every
compromise the data forced rather than taking the panel on trust.
"""

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from headroom.types import as_date

#: Dated regime markers. These are the shifts the project is built to be judged on, so
#: they are named here rather than discovered: the coverage chart is plotted through
#: them and `docs/methods.md` states that they were known in advance and not selected
#: after seeing which method won.
REGIMES: Final[dict[str, date]] = {
    "sandy": date(2012, 10, 29),
    "covid": date(2020, 3, 1),
}

#: Scale factor making the median absolute deviation an estimate of the standard
#: deviation for normally distributed data.
_MAD_TO_SIGMA: Final[float] = 1.4826


@dataclass(frozen=True, slots=True)
class Disagreement:
    """How often the incident borough and the dispatch area disagree.

    Attributes:
        moved_incidents: Incidents whose recorded borough is not the one their dispatch
            area was assigned to.
        total_incidents: Incidents in the frame.
        share: ``moved_incidents / total_incidents``.
        worst_areas: Up to ten areas with the most moved incidents, largest first, as
            ``(area, assigned_borough, moved)``.
    """

    moved_incidents: int
    total_incidents: int
    share: float
    worst_areas: tuple[tuple[str, str, int], ...]


def borough_disagreement(counts: pl.DataFrame, areas: pl.DataFrame) -> Disagreement:
    """Measure the cost of assigning each dispatch area to one borough.

    :func:`headroom.data.nyc_ems.assign_areas_to_boroughs` makes a judgement call. This
    puts a number on it, which is the only thing that makes the call defensible.

    Args:
        counts: The loader's frame, with the per-incident borough.
        areas: The area-to-borough assignment.

    Returns:
        The disagreement report.
    """
    joined = counts.join(areas.rename({"borough": "assigned"}), on="area", how="inner")
    total = int(joined["n"].sum())
    moved = joined.filter(pl.col("borough") != pl.col("assigned"))
    worst = (
        moved.group_by("area", "assigned")
        .agg(pl.col("n").sum())
        .sort("n", descending=True)
        .head(10)
    )
    moved_n = int(moved["n"].sum())
    return Disagreement(
        moved_incidents=moved_n,
        total_incidents=total,
        share=moved_n / total if total else 0.0,
        worst_areas=tuple((a, b, int(n)) for a, b, n in worst.rows()),
    )


def missing_days(counts: pl.DataFrame) -> tuple[date, ...]:
    """Return days inside the record on which the dataset holds nothing at all.

    A quiet area on a given day is a zero and is filled as one. A day with no rows for
    *any* area is the dataset missing, which is a different thing and must not be filled
    silently.

    Args:
        counts: Any frame with a ``day`` column.

    Returns:
        The missing days, ascending.

    Raises:
        ValueError: The frame is empty.
    """
    if counts.height == 0:
        raise ValueError("no rows to check")
    lo = as_date(counts["day"].min(), "first day")
    hi = as_date(counts["day"].max(), "last day")
    present = set(counts["day"].unique().to_list())
    full = pl.date_range(lo, hi, interval="1d", eager=True).to_list()
    return tuple(day for day in full if day not in present)


def coverage(counts: pl.DataFrame) -> pl.DataFrame:
    """Report how much of the record each dispatch area is active for.

    Args:
        counts: One row per area per day.

    Returns:
        A frame of ``area``, ``first``, ``last``, ``active_days``, ``share`` and
        ``incidents``, sorted by share ascending so the thin series come first.
    """
    lo = as_date(counts["day"].min(), "first day")
    hi = as_date(counts["day"].max(), "last day")
    span = (hi - lo).days + 1
    return (
        counts.group_by("area")
        .agg(
            pl.col("day").min().alias("first"),
            pl.col("day").max().alias("last"),
            pl.col("day").n_unique().alias("active_days"),
            pl.col("n").sum().alias("incidents"),
        )
        .with_columns((pl.col("active_days") / span).alias("share"))
        .sort("share", "area")
    )


def outliers(
    values: npt.NDArray[np.float64], days: tuple[date, ...], threshold: float = 6.0
) -> tuple[tuple[date, float, float], ...]:
    """Flag days far from a robust centre, as ``(day, value, robust z)``.

    Median and MAD rather than mean and standard deviation, because the events worth
    finding here are large enough to drag a mean towards themselves and hide.

    Nothing is removed. A one-day spike in emergency demand is the signal this project
    exists to forecast through; the flag exists so `docs/data.md` can name the dates and
    say what they were.

    Args:
        values: One series, shape ``(n_days,)``.
        days: The dates, same length.
        threshold: Robust z above which a day is flagged.

    Returns:
        The flagged days, largest absolute z first.

    Raises:
        ValueError: The series and the dates are different lengths.
    """
    if len(values) != len(days):
        raise ValueError(f"{len(values)} values, {len(days)} days")
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    # A series that is constant apart from its spikes has a MAD of zero, and dividing by
    # it would report no outliers on the one shape where they are most obvious. Fall back
    # to the standard deviation there; only a series with no variation at all returns none.
    scale = _MAD_TO_SIGMA * mad if mad > 0.0 else float(np.std(values))
    if scale == 0.0:
        return ()
    z = (values - centre) / scale
    hits = np.flatnonzero(np.abs(z) >= threshold)
    order = hits[np.argsort(-np.abs(z[hits]))]
    return tuple((days[i], float(values[i]), float(z[i])) for i in order)
