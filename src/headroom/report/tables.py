"""Format measured numbers into the README's tables, and write them into the README.

Nothing here computes a score. The command that does (`headroom report`) hands over
point estimates with their intervals, and this module decides only how they read. Kept
apart so the formatting and the one place the README is written can be tested without a
backtest.

## The markers

The README holds a start and an end marker around its results. :func:`splice` replaces
what lies between them and leaves every other line untouched, and it refuses to write
when the markers are missing, repeated or out of order, because guessing where the tables
go would overwrite prose someone wrote by hand.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

START: Final[str] = "<!-- report:start (written by `headroom report`; do not edit by hand) -->"
END: Final[str] = "<!-- report:end -->"


@dataclass(frozen=True, slots=True)
class Estimate:
    """A measured number with its 95 percent interval.

    Attributes:
        point: The estimate on the data itself.
        low: Lower end of the interval.
        high: Upper end of the interval.
    """

    point: float
    low: float
    high: float

    @classmethod
    def of(cls, triple: tuple[float, float, float]) -> "Estimate":
        """Build from the ``(point, low, high)`` the bootstrap functions return.

        Args:
            triple: ``(point, low, high)``.

        Returns:
            The estimate.
        """
        return cls(*triple)


def decimals_for(value: float) -> int:
    """Choose decimals so a number shows three significant figures, at most three places.

    Args:
        value: A typical magnitude for the column, such as the point estimate.

    Returns:
        Places after the decimal point: 0 from 100 upward, 1 from 10, 2 from 1, else 3.
    """
    size = abs(value)
    if size >= 100:
        return 0
    if size >= 10:
        return 1
    if size >= 1:
        return 2
    return 3


def number(estimate: Estimate, places: int | None = None, signed: bool = False) -> str:
    """Format an estimate as ``point [low, high]``, all three to the same places.

    Args:
        estimate: The estimate.
        places: Decimal places; chosen from the point estimate if omitted.
        signed: Show a plus sign on positive values, for skills and differences.

    Returns:
        For example ``+0.184 [+0.170, +0.197]``.
    """
    digits = decimals_for(estimate.point) if places is None else places
    sign = "+" if signed else ""
    # Rounded first, and a rounded zero made positive, so a tie never reads as "-0.00".
    values = [round(v, digits) + 0.0 for v in (estimate.point, estimate.low, estimate.high)]
    shown = [f"{v:{sign}.{digits}f}" for v in values]
    return f"{shown[0]} [{shown[1]}, {shown[2]}]"


def duration(seconds: float) -> str:
    """Format compute time: minutes under an hour, hours to one place above it.

    Args:
        seconds: Seconds of compute.

    Returns:
        For example ``4 min`` or ``18.3 h``.

    Raises:
        ValueError: The time is negative or not finite.
    """
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"compute time must be a non-negative number, got {seconds}")
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} min"
    return f"{seconds / 3600:.1f} h"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a Markdown table.

    Args:
        header: Column names.
        rows: Cells, each row as long as the header.

    Returns:
        The table, one line per row, without a trailing newline.

    Raises:
        ValueError: A row's length differs from the header's, or a cell contains a pipe.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f"row {list(row)} has {len(row)} cells, header has {len(header)}")
        if any("|" in cell for cell in row):
            raise ValueError(f"a cell contains a pipe, which would break the table: {row}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def worst_window(per_origin: npt.NDArray[np.float64], window: int) -> tuple[int, float]:
    """Return where a trailing rolling mean is lowest, and its value.

    The coverage column reports the worst the interval ever got. The window is trailing,
    for the reason :func:`headroom.score.coverage.rolling_coverage` gives, and positions
    before it is full are not candidates.

    Args:
        per_origin: One coverage fraction per origin, in origin order.
        window: Window length in origins.

    Returns:
        ``(index, value)`` for the window ending at ``index``.

    Raises:
        ValueError: The window is not positive or is longer than the series.
    """
    if per_origin.ndim != 1:
        raise ValueError("the worst window takes one series at a time")
    if not 1 <= window <= per_origin.size:
        raise ValueError(f"window {window} does not fit {per_origin.size} origins")
    cumulative = np.concatenate(([0.0], np.cumsum(per_origin)))
    rolling = (cumulative[window:] - cumulative[:-window]) / window
    at = int(np.argmin(rolling))
    return at + window - 1, float(rolling[at])


def splice(text: str, block: str) -> str:
    """Replace what lies between the report markers with a new block.

    Args:
        text: The README.
        block: The generated tables and notes, without the markers.

    Returns:
        The README with the block in place and everything outside the markers unchanged.

    Raises:
        ValueError: The markers are missing, repeated, or the end comes before the start.
    """
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(
            f"the README must hold exactly one of each report marker, {START!r} and "
            f"{END!r}; add them where the tables belong"
        )
    head, rest = text.split(START)
    if END not in rest:
        raise ValueError("the report end marker comes before the start marker")
    _, tail = rest.split(END)
    return f"{head}{START}\n{block.strip()}\n{END}{tail}"
