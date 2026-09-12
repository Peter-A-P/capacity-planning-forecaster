"""Calendar features: day of week, and the US federal holidays.

The holidays are computed from their statutory rules rather than taken from a package.
Eleven rules are less code than a dependency wrapper, they are testable against dates
anyone can check, and pinning a holiday package for a table that must reproduce in 2030
buys a risk for no benefit.

Demand is dated by when the incident happened, so holidays are placed on the day the
holiday falls, not the day a federal office observes it. Both are available: the
observed shift matters for a call centre and is kept for whoever substitutes their own
series, and `docs/data.md` says which the tables use.
"""

from datetime import date, timedelta
from typing import Final

import polars as pl

#: Juneteenth became a federal holiday in 2021. Before that it is not one, and treating
#: it as one for twenty years of history would put a step in a calendar feature.
JUNETEENTH_FROM: Final[int] = 2021

_MONDAY: Final[int] = 0
_THURSDAY: Final[int] = 3
_SATURDAY: Final[int] = 5
_SUNDAY: Final[int] = 6


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth given weekday of a month.

    Args:
        year: Calendar year.
        month: Month, 1 to 12.
        weekday: Monday is 0, as in :meth:`datetime.date.weekday`.
        n: Which one, counting from 1.

    Returns:
        The date.
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last given weekday of a month.

    Args:
        year: Calendar year.
        month: Month, 1 to 12.
        weekday: Monday is 0.

    Returns:
        The date.
    """
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def federal_holidays(year: int) -> dict[date, str]:
    """Return the US federal holidays falling in a year, by actual date.

    Args:
        year: Calendar year.

    Returns:
        A mapping from date to holiday name.
    """
    holidays: dict[date, str] = {
        date(year, 1, 1): "new_year",
        _nth_weekday(year, 1, _MONDAY, 3): "mlk",
        _nth_weekday(year, 2, _MONDAY, 3): "washington",
        _last_weekday(year, 5, _MONDAY): "memorial",
        date(year, 7, 4): "independence",
        _nth_weekday(year, 9, _MONDAY, 1): "labor",
        _nth_weekday(year, 10, _MONDAY, 2): "columbus",
        date(year, 11, 11): "veterans",
        _nth_weekday(year, 11, _THURSDAY, 4): "thanksgiving",
        date(year, 12, 25): "christmas",
    }
    if year >= JUNETEENTH_FROM:
        holidays[date(year, 6, 19)] = "juneteenth"
    return holidays


def observed(day: date) -> date:
    """Return the day a federal office observes a holiday falling on ``day``.

    Saturday moves to the Friday before, Sunday to the Monday after.

    Args:
        day: The holiday's actual date.

    Returns:
        The observed date.
    """
    if day.weekday() == _SATURDAY:
        return day - timedelta(days=1)
    if day.weekday() == _SUNDAY:
        return day + timedelta(days=1)
    return day


def features(days: tuple[date, ...]) -> pl.DataFrame:
    """Build the calendar features for a run of days.

    Args:
        days: The modelling window, ascending.

    Returns:
        A frame of ``day``, ``dow`` (Monday 0), ``is_weekend``, ``holiday`` (the name or
        null), ``is_holiday``, and ``is_holiday_observed``.

    Raises:
        ValueError: ``days`` is empty.
    """
    if not days:
        raise ValueError("no days")
    years = range(days[0].year, days[-1].year + 1)
    actual: dict[date, str] = {}
    for year in years:
        actual.update(federal_holidays(year))
    shifted = {observed(d): name for d, name in actual.items()}

    return pl.DataFrame({"day": list(days)}).with_columns(
        pl.col("day").dt.weekday().sub(1).alias("dow"),
        pl.col("day").dt.weekday().is_in([6, 7]).alias("is_weekend"),
        pl.col("day")
        .replace_strict(actual, default=None, return_dtype=pl.String)
        .alias("holiday"),
        pl.col("day").is_in(list(actual)).alias("is_holiday"),
        pl.col("day").is_in(list(shifted)).alias("is_holiday_observed"),
    )
