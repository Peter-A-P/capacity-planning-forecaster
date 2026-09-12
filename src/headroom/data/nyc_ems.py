"""NYC emergency medical dispatch incidents, aggregated to daily counts.

Source: NYC Open Data, "EMS Incident Dispatch Data" (dataset ``76xm-jjuj``), roughly
thirty million timestamped incidents from 2005 with a borough and a dispatch area on
each one. Licence and terms are recorded in `docs/data.md`.

The aggregation runs on Socrata rather than here. Downloading thirty million rows to
count them would take hours and several gigabytes to reach the same daily counts that
one grouped query returns in seconds, and PLAN.md commits only to aggregating before
anything is stored, not to doing the arithmetic locally. What is cached on disk is the
grouped response, one file per year, which is what a re-run and a checksum need.

The only judgement call in this module is :func:`assign_areas_to_boroughs`. See its
docstring: the borough column is per incident and disagrees with the dispatch area on a
small number of rows, and the hierarchy needs each area to sit under exactly one
borough.
"""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl

#: Socrata host and dataset. Both appear in `docs/data.md`.
DOMAIN: Final[str] = "data.cityofnewyork.us"
DATASET: Final[str] = "76xm-jjuj"

#: Socrata caps a page at 50,000 rows. A single year groups to about 13,000, so the page
#: size is only ever reached if the dispatch areas multiply; the fetch pages anyway.
PAGE: Final[int] = 50_000

#: The first and last days the dataset covers, as published. The loader refuses a window
#: outside this range rather than silently returning a short series.
FIRST_DAY: Final[date] = date(2005, 1, 1)

#: Socrata's own name for Staten Island, normalised on the way in.
_BOROUGH_RENAMES: Final[dict[str, str]] = {
    "RICHMOND / STATEN ISLAND": "STATEN ISLAND",
}

#: Rows whose borough is one of these carry no location and are dropped, with the count
#: reported by :func:`fetch_daily_counts` so the loss is visible rather than assumed.
_UNUSABLE_BOROUGHS: Final[frozenset[str]] = frozenset({"UNKNOWN", ""})


class FetchError(RuntimeError):
    """Socrata returned something the loader cannot use."""


def _query(start: date, end: date, offset: int) -> dict[str, str]:
    """Build the SoQL query for one page of one window.

    Args:
        start: First day of the window, inclusive.
        end: First day after the window, exclusive.
        offset: Row offset within the window's grouped result.

    Returns:
        Query parameters for the Socrata JSON endpoint.
    """
    return {
        "$select": (
            "date_trunc_ymd(incident_datetime) as day, borough, "
            "incident_dispatch_area as area, count(*) as n"
        ),
        "$group": "day, borough, area",
        "$where": (
            f'incident_datetime >= "{start.isoformat()}" '
            f'and incident_datetime < "{end.isoformat()}"'
        ),
        "$order": "day, borough, area",
        "$limit": str(PAGE),
        "$offset": str(offset),
    }


def _year_windows(start: date, end: date) -> Iterator[tuple[int, date, date]]:
    """Split a date range into calendar-year windows.

    One year per request keeps the server-side group small enough to answer in seconds
    and gives the cache a natural key: a completed year never has to be fetched again.

    Args:
        start: First day, inclusive.
        end: First day after the range, exclusive.

    Yields:
        ``(year, window_start, window_end)`` triples covering the range in order.
    """
    for year in range(start.year, end.year + 1):
        lo = max(start, date(year, 1, 1))
        hi = min(end, date(year + 1, 1, 1))
        if lo < hi:
            yield year, lo, hi


def _fetch_window(
    client: httpx.Client, start: date, end: date, app_token: str | None
) -> list[dict[str, Any]]:
    """Fetch every page of one window's grouped counts.

    Args:
        client: An open HTTP client.
        start: First day, inclusive.
        end: First day after the window, exclusive.
        app_token: Optional Socrata application token; raises the anonymous rate limit.

    Returns:
        The grouped rows, as Socrata returned them.

    Raises:
        FetchError: The endpoint returned a non-JSON body or an error payload.
    """
    headers = {"X-App-Token": app_token} if app_token else {}
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = client.get(
            f"https://{DOMAIN}/resource/{DATASET}.json",
            params=_query(start, end, offset),
            headers=headers,
            timeout=180.0,
        )
        response.raise_for_status()
        try:
            page = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - network shape
            raise FetchError(f"{start} to {end}: response was not JSON") from exc
        if not isinstance(page, list):  # pragma: no cover - network shape
            raise FetchError(f"{start} to {end}: {page}")
        rows.extend(page)
        if len(page) < PAGE:
            return rows
        offset += PAGE


def fetch_daily_counts(
    start: date,
    end: date,
    cache_dir: Path,
    app_token: str | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Fetch daily incident counts by borough and dispatch area.

    A calendar year is cached once it has been fetched. The final, partial year is
    refetched every time, because it grows.

    Args:
        start: First day, inclusive. Must be on or after :data:`FIRST_DAY`.
        end: First day after the range, exclusive.
        cache_dir: Directory for the per-year grouped responses. Created if absent.
        app_token: Optional Socrata application token.
        refresh: Refetch every year, ignoring the cache.

    Returns:
        A frame of ``day`` (Date), ``borough``, ``area``, ``n`` (UInt32), sorted, with
        boroughs normalised and unusable rows dropped.

    Raises:
        ValueError: The window is empty or starts before the dataset does.
    """
    if start < FIRST_DAY:
        raise ValueError(f"dataset starts {FIRST_DAY}, asked for {start}")
    if end <= start:
        raise ValueError(f"empty window: {start} to {end}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw: list[dict[str, Any]] = []
    with httpx.Client(follow_redirects=True) as client:
        for year, lo, hi in _year_windows(start, end):
            complete = hi == date(year + 1, 1, 1)
            path = cache_dir / f"{DATASET}-{year}.json"
            if path.exists() and complete and not refresh:
                raw.extend(json.loads(path.read_text(encoding="utf-8")))
                continue
            rows = _fetch_window(client, lo, hi, app_token)
            if complete:
                path.write_text(json.dumps(rows), encoding="utf-8")
            raw.extend(rows)

    return _tidy(raw)


def _tidy(raw: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn Socrata's grouped JSON into the loader's frame.

    Args:
        raw: Grouped rows as returned by Socrata; every value is a string.

    Returns:
        A sorted frame of ``day``, ``borough``, ``area``, ``n``.
    """
    frame = pl.DataFrame(
        raw,
        schema={"day": pl.String, "borough": pl.String, "area": pl.String, "n": pl.String},
    )
    if frame.height == 0:
        return pl.DataFrame(
            schema={"day": pl.Date, "borough": pl.String, "area": pl.String, "n": pl.UInt32}
        )
    return (
        frame.with_columns(
            pl.col("day").str.slice(0, 10).str.to_date(),
            pl.col("borough").fill_null("").str.strip_chars().str.to_uppercase(),
            pl.col("area").fill_null("").str.strip_chars().str.to_uppercase(),
            pl.col("n").cast(pl.UInt32),
        )
        .with_columns(pl.col("borough").replace(_BOROUGH_RENAMES))
        .filter(~pl.col("borough").is_in(list(_UNUSABLE_BOROUGHS)) & (pl.col("area") != ""))
        # A window boundary can split one day across two cached years only if the caller
        # asked for a mid-year start, so sum rather than assume the group is already unique.
        .group_by("day", "borough", "area")
        .agg(pl.col("n").sum())
        .sort("day", "borough", "area")
    )


def assign_areas_to_boroughs(counts: pl.DataFrame) -> pl.DataFrame:
    """Assign each dispatch area to exactly one borough.

    The dataset carries a borough on each *incident*, and on a small number of incidents
    it disagrees with the dispatch area: a handful of rows a day put area ``M9`` in the
    Bronx or ``K2`` in Manhattan, almost always with a count of one. A dispatch area is a
    fixed piece of geography, so those are attribute errors on individual incidents, not
    an area that moved.

    The hierarchy needs each area under exactly one borough or the summing matrix is not
    a matrix. Each area is therefore assigned to the borough it is recorded under most
    often across the whole history, and :func:`headroom.data.checks.borough_disagreement`
    reports what share of incidents that moved, so the size of the judgement is published
    rather than hidden.

    Args:
        counts: The loader's frame, from :func:`fetch_daily_counts`.

    Returns:
        A frame of ``area`` and the ``borough`` it belongs to, sorted by area.
    """
    return (
        counts.group_by("area", "borough")
        .agg(pl.col("n").sum())
        .sort("area", "n", "borough", descending=[False, True, False])
        .group_by("area", maintain_order=True)
        .first()
        .select("area", "borough")
        .sort("area")
    )


def daily_leaf_counts(counts: pl.DataFrame, areas: pl.DataFrame) -> pl.DataFrame:
    """Collapse the loader's frame to one count per dispatch area per day.

    The incident-level borough is dropped here in favour of the area's assigned borough,
    which is what makes the hierarchy coherent.

    Args:
        counts: The loader's frame, from :func:`fetch_daily_counts`.
        areas: The area-to-borough assignment, from :func:`assign_areas_to_boroughs`.

    Returns:
        A frame of ``day``, ``borough``, ``area``, ``n``, one row per area per day,
        sorted.
    """
    return (
        counts.drop("borough")
        .join(areas, on="area", how="inner")
        .group_by("day", "borough", "area")
        .agg(pl.col("n").sum())
        .select("day", "borough", "area", "n")
        .sort("day", "area")
    )
