"""Build a hierarchy and its panel from the loader's daily counts.

Two decisions live here and both are reported rather than assumed.

**Which areas are leaves.** An area has to pass two tests, because two different things
disqualify one.

Dispatch areas open and close over twenty-one years, and a series that exists for a
third of the record cannot be backtested over the whole of it. So an area is kept only
if the span from its first incident to its last covers at least :data:`MIN_COVERAGE` of
the modelling window. Span rather than a count of active days, because a small area with
no incident on a quiet Tuesday recorded a zero, not an absence, and counting active days
would drop it for being quiet.

Span alone is not enough. A stray code that appears once or twice a year for twenty
years spans the whole window and is still not a dispatch area, and forecasting a series
that is 96 percent zeros would put a meaningless row in every table. So an area must
also average at least :data:`MIN_DAILY` incidents a day.

The dropped areas' share of all incidents is returned, and `docs/data.md` prints it.

**What the root means once areas are dropped.** The city series is the sum of the
retained leaves, not the sum of every incident. Defining it any other way would make the
hierarchy incoherent before a single forecast was made, and a reconciliation result on an
incoherent hierarchy is not a result. The difference is the same dropped share.
"""

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from headroom.hierarchy.spec import Hierarchy, from_paths
from headroom.types import as_date

#: An area's first-to-last span must cover at least this share of the window to be a
#: leaf. 0.98 keeps every area that ran for the whole period and drops the ones that
#: opened or closed inside it; the distribution has a clear gap, shown in `docs/data.md`.
MIN_COVERAGE: Final[float] = 0.98

#: An area must average at least this many incidents a day to be a leaf. On the NYC
#: record the gap is two orders of magnitude wide, between two stray codes at 0.4 a day
#: and the quietest real dispatch area at 20.6, so any threshold between them selects the
#: same leaves and this one is not a tuned number. `docs/data.md` prints the distribution.
MIN_DAILY: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class Panel:
    """Daily values for every node of a hierarchy.

    Attributes:
        hierarchy: The structure the values belong to.
        days: The modelling window, one entry per row of :attr:`values`, contiguous and
            ascending with no gaps.
        values: Node values, shape ``(n_nodes, n_days)``. Coherent by construction: the
            aggregates are built from the leaves.
        dropped_areas: Areas excluded for thin coverage, sorted.
        dropped_share: Those areas' share of all incidents in the window.
    """

    hierarchy: Hierarchy
    days: tuple[date, ...]
    values: npt.NDArray[np.float64]
    dropped_areas: tuple[str, ...]
    dropped_share: float

    def __post_init__(self) -> None:
        """Check the panel's shape against its hierarchy.

        Raises:
            ValueError: The value matrix does not match the hierarchy or the days.
        """
        expected = (self.hierarchy.n_nodes, len(self.days))
        if self.values.shape != expected:
            raise ValueError(f"values are {self.values.shape}, expected {expected}")

    def leaf_values(self) -> npt.NDArray[np.float64]:
        """Return the leaf rows of :attr:`values`."""
        return self.values[-self.hierarchy.n_leaves :]

    def series(self, node: str) -> npt.NDArray[np.float64]:
        """Return one node's series through time.

        Args:
            node: The node identifier.

        Returns:
            The daily values, shape ``(n_days,)``.
        """
        row: npt.NDArray[np.float64] = self.values[self.hierarchy.index(node)]
        return row


def build(
    leaf_counts: pl.DataFrame,
    start: date | None = None,
    end: date | None = None,
    min_coverage: float = MIN_COVERAGE,
    min_daily: float = MIN_DAILY,
) -> Panel:
    """Build the hierarchy and its daily panel.

    Days on which a retained area recorded nothing are filled with zero. That is a real
    count, not a gap: a dispatch area with no incident on a quiet night is a zero, and
    :func:`headroom.data.checks.missing_days` is what catches a day the dataset is
    actually missing.

    Args:
        leaf_counts: One row per area per day, from
            :func:`headroom.data.nyc_ems.daily_leaf_counts`.
        start: First day of the modelling window. Defaults to the first day present.
        end: Last day of the modelling window, inclusive. Defaults to the last present.
        min_coverage: Share of the window an area's first-to-last span must cover.
        min_daily: Mean incidents a day an area must reach.

    Returns:
        The panel, with the hierarchy inside it.

    Raises:
        ValueError: The window is empty, or every area was dropped.
    """
    lo = start or as_date(leaf_counts["day"].min(), "first day present")
    hi = end or as_date(leaf_counts["day"].max(), "last day present")
    if hi < lo:
        raise ValueError(f"empty window: {lo} to {hi}")

    window = leaf_counts.filter(pl.col("day").is_between(lo, hi))
    days = pl.date_range(lo, hi, interval="1d", eager=True)
    n_days = days.len()

    per_area = window.group_by("area").agg(
        ((pl.col("day").max() - pl.col("day").min()).dt.total_days() + 1).alias("span"),
        (pl.col("n").sum() / n_days).alias("per_day"),
    )
    keep = (pl.col("span") >= min_coverage * n_days) & (pl.col("per_day") >= min_daily)
    kept = per_area.filter(keep)["area"].sort().to_list()
    dropped = per_area.filter(~keep)["area"].sort().to_list()
    if not kept:
        raise ValueError(
            f"no area both spans {min_coverage:.0%} of the {n_days}-day window "
            f"and averages {min_daily} incidents a day"
        )

    total = int(window["n"].sum())
    dropped_n = int(window.filter(pl.col("area").is_in(dropped))["n"].sum())

    retained = window.filter(pl.col("area").is_in(kept))
    pairs = retained.select("borough", "area").unique().sort("borough", "area").rows()
    hierarchy = from_paths([(b, a) for b, a in pairs])

    grid = (
        pl.DataFrame({"day": days})
        .join(retained.select("area").unique(), how="cross")
        .join(retained.select("day", "area", "n"), on=["day", "area"], how="left")
        .with_columns(pl.col("n").fill_null(0))
    )
    wide = grid.pivot(on="area", index="day", values="n", aggregate_function="sum").sort("day")
    # Column order follows the hierarchy's leaf order, not the pivot's.
    leaf_area = [leaf.rsplit("/", 1)[1] for leaf in hierarchy.leaves]
    leaf_values = wide.select(leaf_area).to_numpy().T.astype(np.float64)

    return Panel(
        hierarchy=hierarchy,
        days=tuple(days.to_list()),
        values=hierarchy.aggregate(leaf_values),
        dropped_areas=tuple(dropped),
        dropped_share=dropped_n / total if total else 0.0,
    )
