"""The dashboard's JSON, and the schema it has to satisfy.

Nothing here computes a score, for the same reason nothing in :mod:`headroom.report.tables`
does. `headroom export` opens the checkpoints, scores them on the same origins the README's
tables use and hands the numbers over; this module decides only how they are shaped, rounded
and checked. That split is what lets the payloads be tested without a backtest.

## Why there is a schema at all

The dashboard is static: the page is served from a CDN with no backend to fail loudly, so a
payload that has lost a field fails in the browser, on someone else's machine, as a blank
panel. :func:`validate` runs before anything is written, so a bad export fails in the
terminal instead. `PLAN.md` section 4 names this as one of the tests that matter.

## Two files, not one

`forecast.json` holds the fan chart's bands for every node at every origin and is the large
one. Everything else (coverage through the shift, the reconciliation table, the staffing
table the service-level slider reads) is small and lives in `dashboard.json`, so the page
can draw its tables before the bands have arrived.

## nan is not JSON

Several of these series are `nan` before a method could calibrate, and `NaN` is not valid
JSON however many parsers accept it. Every missing value becomes `null`, and the writer
passes ``allow_nan=False`` so that a missed conversion raises here rather than producing a
file that only some parsers read.
"""

import json
import math
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

#: Bumped when a payload changes shape in a way the page has to know about. The page checks
#: it and says so rather than drawing half a chart from a file it does not understand.
SCHEMA_VERSION: Final[int] = 1

#: The quantiles the fan chart draws, ascending. Five, not the nine the PNG uses: every level
#: is one more array per node per origin in a file the browser downloads, and the outer band
#: plus the quartiles is what a planner reads off a fan.
FAN_LEVELS: Final[tuple[float, ...]] = (0.05, 0.25, 0.5, 0.75, 0.95)

#: Decimal places for demand and cost. Incidents and units are counted in whole numbers, so
#: one place is already finer than the quantity it describes; it is kept because a dispatch
#: area's daily count is small enough that rounding to an integer would be visible.
DEMAND_PLACES: Final[int] = 1

#: Decimal places for a share, a coverage or a skill.
SHARE_PLACES: Final[int] = 4

#: The file names, which are also the schema names.
DASHBOARD: Final[str] = "dashboard.json"
FORECAST: Final[str] = "forecast.json"


def number(value: float, places: int) -> float | None:
    """Round one value for JSON, turning a missing one into ``None``.

    Args:
        value: The value.
        places: Decimal places.

    Returns:
        The rounded value, or ``None`` where it is not finite.
    """
    return None if not math.isfinite(value) else round(float(value), places)


def series(values: npt.NDArray[np.float64], places: int) -> list[float | None]:
    """Round a series for JSON.

    Args:
        values: One value per point, possibly with ``nan`` where nothing was measured.
        places: Decimal places.

    Returns:
        The rounded values, ``None`` where the input was not finite.
    """
    return [number(float(value), places) for value in np.asarray(values).ravel()]


def estimate(triple: tuple[float, float, float], places: int) -> dict[str, float | None]:
    """Shape a measured number with its interval.

    Args:
        triple: ``(point, low, high)``, as the bootstrap functions return it.
        places: Decimal places.

    Returns:
        The three values under ``point``, ``low`` and ``high``.

    Raises:
        ValueError: The interval does not contain its own point estimate, which means the
            bootstrap was run on too short a window to say anything (`docs/methods.md`).
    """
    point, low, high = triple
    if math.isfinite(point) and not low <= point <= high:
        raise ValueError(f"interval [{low:g}, {high:g}] does not contain its point {point:g}")
    return {
        "point": number(point, places),
        "low": number(low, places),
        "high": number(high, places),
    }


def _days(days: Iterable[date]) -> list[str]:
    """Format dates the way the page parses them.

    Args:
        days: The dates.

    Returns:
        ISO strings.
    """
    return [day.isoformat() for day in days]


def dashboard_payload(
    *,
    generated: date,
    origins: dict[str, Any],
    hierarchy: dict[str, Any],
    coverage: dict[str, Any],
    reconciliation: dict[str, Any],
    staffing: dict[str, Any],
) -> dict[str, Any]:
    """Assemble everything the page needs except the fan bands.

    Args:
        generated: The day the export ran.
        origins: As :func:`origins_section` returns it.
        hierarchy: As :func:`hierarchy_section` returns it.
        coverage: As :func:`coverage_section` returns it.
        reconciliation: As :func:`reconciliation_section` returns it.
        staffing: As :func:`staffing_section` returns it.

    Returns:
        The payload.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": generated.isoformat(),
        "origins": origins,
        "hierarchy": hierarchy,
        "coverage": coverage,
        "reconciliation": reconciliation,
        "staffing": staffing,
    }


def origins_section(
    *, days: Sequence[date], step: int, horizon: int, train_window: int | None
) -> dict[str, Any]:
    """Describe the schedule every number on the page was measured on.

    Args:
        days: The day of each scored origin, in order.
        step: Days between origins.
        horizon: Days each origin forecasts.
        train_window: Trailing training window in days, or ``None`` if it expands.

    Returns:
        The section.

    Raises:
        ValueError: There are no origins.
    """
    if not days:
        raise ValueError("no origins to export")
    return {
        "count": len(days),
        "first": days[0].isoformat(),
        "last": days[-1].isoformat(),
        "step_days": step,
        "horizon_days": horizon,
        "train_window_days": train_window,
    }


def hierarchy_section(*, nodes: Sequence[str], levels: Sequence[str]) -> dict[str, Any]:
    """Describe the series, so the page can offer them without guessing at identifiers.

    Args:
        nodes: Node identifiers, in the hierarchy's row order.
        levels: Each node's level, in the same order.

    Returns:
        The section.

    Raises:
        ValueError: The two sequences disagree in length.
    """
    if len(nodes) != len(levels):
        raise ValueError(f"{len(nodes)} nodes and {len(levels)} levels")
    return {
        "nodes": list(nodes),
        "levels": list(levels),
        "counts": {level: levels.count(level) for level in dict.fromkeys(levels)},
    }


def coverage_section(
    *,
    level: str,
    days: Sequence[date],
    window_origins: int,
    shift: tuple[date, date],
    methods: Sequence[tuple[str, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]],
) -> dict[str, Any]:
    """Shape the coverage chart's series.

    The values are per origin and unsmoothed. The page takes the trailing mean itself, with
    the rule :func:`headroom.report.charts.rolling_mean` uses, so that the window can be
    changed on the page without re-exporting and the PNG in the README stays the reference
    for the default one.

    Args:
        level: The hierarchy level coverage is measured at.
        days: The day of each origin.
        window_origins: Origins in the default trailing window.
        shift: The period the page marks as the demand shift.
        methods: One tuple per line: name, nominal coverage, coverage per origin, mean
            width per origin.

    Returns:
        The section.

    Raises:
        ValueError: A series is not one value per origin, or there are no series.
    """
    if not methods:
        raise ValueError("no coverage series to export")
    lines = []
    for name, nominal, covered, width in methods:
        if covered.size != len(days) or width.size != len(days):
            raise ValueError(
                f"{name} at {nominal:.0%}: {covered.size} coverage and {width.size} width "
                f"values for {len(days)} origins"
            )
        lines.append(
            {
                "method": name,
                "nominal": round(float(nominal), SHARE_PLACES),
                "coverage": series(covered, SHARE_PLACES),
                "width": series(width, DEMAND_PLACES),
            }
        )
    return {
        "level": level,
        "days": _days(days),
        "window_origins": window_origins,
        "shift": [shift[0].isoformat(), shift[1].isoformat()],
        "series": lines,
    }


def reconciliation_section(
    *,
    model: str,
    base_coherence_error: float,
    reconciled_coherence_error: float,
    negative_share: float,
    nominal: float,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Shape the reconciliation view.

    Args:
        model: The model that was reconciled.
        base_coherence_error: Largest breach of the summing constraints before MinT.
        reconciled_coherence_error: The same after it, over every draw.
        negative_share: Share of reconciled path values below zero.
        nominal: The nominal coverage the coverage columns report.
        rows: One row per level, as :func:`reconciliation_row` returns it.

    Returns:
        The section.

    Raises:
        ValueError: There are no rows.
    """
    if not rows:
        raise ValueError("no reconciliation rows to export")
    return {
        "model": model,
        "base_coherence_error": float(base_coherence_error),
        "reconciled_coherence_error": float(reconciled_coherence_error),
        "negative_share": round(float(negative_share), 6),
        "nominal": round(float(nominal), SHARE_PLACES),
        "rows": list(rows),
    }


def reconciliation_row(
    *,
    level: str,
    base_crps: tuple[float, float, float],
    variants: Sequence[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
) -> dict[str, Any]:
    """Shape one level's row of the reconciliation table.

    Args:
        level: The hierarchy level.
        base_crps: CRPS before reconciling, with its interval.
        variants: One tuple per reconciliation: its name, the change in CRPS against the
            base, and its coverage at the nominal level.

    Returns:
        The row.
    """
    places = 3
    return {
        "level": level,
        "crps": estimate(base_crps, places),
        "variants": [
            {
                "name": name,
                "crps_change": estimate(change, places),
                "coverage": estimate(covered, SHARE_PLACES),
            }
            for name, change, covered in variants
        ],
    }


def staffing_section(
    *,
    demand_per_unit: float,
    cost_over: float,
    cost_under: float,
    implied: float,
    oracle_units: float,
    reference: str,
    service_levels: Sequence[float],
    methods: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Shape the staffing table the service-level slider reads.

    Args:
        demand_per_unit: Incidents one staffed unit serves in a day.
        cost_over: Cost of a unit-day staffed and not needed.
        cost_under: Cost of a unit-day needed and not staffed.
        implied: The service level those costs imply.
        oracle_units: Units a day an oracle knowing demand would staff.
        reference: The method the cost differences are taken against.
        service_levels: The levels staffed at, ascending; the slider's stops.
        methods: One entry per method, as :func:`staffing_method` returns it.

    Returns:
        The section.

    Raises:
        ValueError: A method does not have one value per service level, or the reference
            method is not among them.
    """
    if reference not in {method["name"] for method in methods}:
        raise ValueError(f"reference method {reference} is not in the staffing table")
    for method in methods:
        for field in ("units", "cost", "cost_change"):
            if len(method[field]) != len(service_levels):
                raise ValueError(
                    f"{method['name']}: {len(method[field])} {field} values for "
                    f"{len(service_levels)} service levels"
                )
    return {
        "inputs": {
            "demand_per_unit": float(demand_per_unit),
            "cost_over": float(cost_over),
            "cost_under": float(cost_under),
            "implied_service_level": round(float(implied), 6),
        },
        "oracle_units": number(oracle_units, DEMAND_PLACES),
        "reference": reference,
        "service_levels": [round(float(level), 6) for level in service_levels],
        "methods": list(methods),
    }


def staffing_method(
    *,
    name: str,
    units: Sequence[tuple[float, float, float]],
    cost: Sequence[tuple[float, float, float]],
    cost_change: Sequence[tuple[float, float, float] | None],
) -> dict[str, Any]:
    """Shape one method's row of the staffing table, at every service level.

    Args:
        name: The method's label.
        units: Units staffed a day at each service level, with intervals.
        cost: Realised cost a day against the oracle, likewise.
        cost_change: Cost against the reference method, or ``None`` at each level for the
            reference method itself.

    Returns:
        The entry.
    """
    return {
        "name": name,
        "units": [estimate(value, DEMAND_PLACES) for value in units],
        "cost": [estimate(value, 2) for value in cost],
        "cost_change": [None if value is None else estimate(value, 2) for value in cost_change],
    }


def forecast_payload(
    *,
    model: str,
    horizon_step: int,
    days: Sequence[date],
    nodes: Sequence[str],
    node_levels: Sequence[str],
    actual: npt.NDArray[np.float64],
    bands: npt.NDArray[np.float64],
    levels: Sequence[float] = FAN_LEVELS,
) -> dict[str, Any]:
    """Shape the fan chart's bands for every node.

    Laid out node-major, ``bands[node][level][day]``, because the page draws one node at a
    time and slicing a node out of that layout is one index rather than a walk over the
    whole array.

    Args:
        model: The model whose forecasts these are.
        horizon_step: Days ahead each forecast was made, between 1 and the horizon.
        days: The target day of each forecast, in order.
        nodes: Node identifiers.
        node_levels: Each node's hierarchy level.
        actual: What happened, shape ``(n_nodes, n_days)``.
        bands: Forecast quantiles, shape ``(n_nodes, n_levels, n_days)``.
        levels: The quantile levels, ascending.

    Returns:
        The payload.

    Raises:
        ValueError: The arrays do not agree with the nodes, levels and days given.
    """
    shape = (len(nodes), len(levels), len(days))
    if bands.shape != shape:
        raise ValueError(f"bands are {bands.shape}, nodes and days say {shape}")
    if actual.shape != (len(nodes), len(days)):
        raise ValueError(f"actual is {actual.shape}, expected {(len(nodes), len(days))}")
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "horizon_step": horizon_step,
        "levels": [round(float(level), 4) for level in levels],
        "days": _days(days),
        "nodes": list(nodes),
        "node_levels": list(node_levels),
        "actual": [series(row, DEMAND_PLACES) for row in actual],
        "bands": [[series(row, DEMAND_PLACES) for row in node] for node in bands],
    }


def _estimate_schema() -> dict[str, Any]:
    """The shape of a measured number with its interval.

    Returns:
        The schema fragment.
    """
    number_or_null = {"type": ["number", "null"]}
    return {
        "type": "object",
        "required": ["point", "low", "high"],
        "additionalProperties": False,
        "properties": {"point": number_or_null, "low": number_or_null, "high": number_or_null},
    }


def _series_schema() -> dict[str, Any]:
    """The shape of a series with gaps in it.

    Returns:
        The schema fragment.
    """
    return {"type": "array", "items": {"type": ["number", "null"]}}


#: What each payload has to be, as JSON Schema. Written out rather than derived from the
#: builders above, so that a builder losing a field is a test failure and not a schema that
#: quietly agrees with it.
SCHEMAS: Final[dict[str, dict[str, Any]]] = {
    DASHBOARD: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Capacity Planning Forecaster dashboard",
        "type": "object",
        "required": [
            "schema_version",
            "generated",
            "origins",
            "hierarchy",
            "coverage",
            "reconciliation",
            "staffing",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "generated": {"type": "string", "format": "date"},
            "origins": {
                "type": "object",
                "required": ["count", "first", "last", "step_days", "horizon_days"],
                "properties": {
                    "count": {"type": "integer", "minimum": 1},
                    "first": {"type": "string"},
                    "last": {"type": "string"},
                    "step_days": {"type": "integer", "minimum": 1},
                    "horizon_days": {"type": "integer", "minimum": 1},
                    "train_window_days": {"type": ["integer", "null"]},
                },
            },
            "hierarchy": {
                "type": "object",
                "required": ["nodes", "levels", "counts"],
                "properties": {
                    "nodes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "levels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "counts": {"type": "object"},
                },
            },
            "coverage": {
                "type": "object",
                "required": ["level", "days", "window_origins", "shift", "series"],
                "properties": {
                    "level": {"type": "string"},
                    "days": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "window_origins": {"type": "integer", "minimum": 1},
                    "shift": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "series": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["method", "nominal", "coverage", "width"],
                            "properties": {
                                "method": {"type": "string"},
                                "nominal": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "coverage": _series_schema(),
                                "width": _series_schema(),
                            },
                        },
                    },
                },
            },
            "reconciliation": {
                "type": "object",
                "required": [
                    "model",
                    "base_coherence_error",
                    "reconciled_coherence_error",
                    "negative_share",
                    "nominal",
                    "rows",
                ],
                "properties": {
                    "model": {"type": "string"},
                    "base_coherence_error": {"type": "number", "minimum": 0},
                    "reconciled_coherence_error": {"type": "number", "minimum": 0},
                    "negative_share": {"type": "number", "minimum": 0, "maximum": 1},
                    "nominal": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "rows": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["level", "crps", "variants"],
                            "properties": {
                                "level": {"type": "string"},
                                "crps": _estimate_schema(),
                                "variants": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "required": ["name", "crps_change", "coverage"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "crps_change": _estimate_schema(),
                                            "coverage": _estimate_schema(),
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "staffing": {
                "type": "object",
                "required": [
                    "inputs",
                    "oracle_units",
                    "reference",
                    "service_levels",
                    "methods",
                ],
                "properties": {
                    "inputs": {
                        "type": "object",
                        "required": [
                            "demand_per_unit",
                            "cost_over",
                            "cost_under",
                            "implied_service_level",
                        ],
                        "properties": {
                            "demand_per_unit": {"type": "number", "exclusiveMinimum": 0},
                            "cost_over": {"type": "number", "exclusiveMinimum": 0},
                            "cost_under": {"type": "number", "exclusiveMinimum": 0},
                            "implied_service_level": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "exclusiveMaximum": 1,
                            },
                        },
                    },
                    "oracle_units": {"type": "number", "minimum": 0},
                    "reference": {"type": "string"},
                    "service_levels": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 1,
                        },
                    },
                    "methods": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["name", "units", "cost", "cost_change"],
                            "properties": {
                                "name": {"type": "string"},
                                "units": {"type": "array", "items": _estimate_schema()},
                                "cost": {"type": "array", "items": _estimate_schema()},
                                "cost_change": {
                                    "type": "array",
                                    "items": {"oneOf": [_estimate_schema(), {"type": "null"}]},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    FORECAST: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Capacity Planning Forecaster fan bands",
        "type": "object",
        "required": [
            "schema_version",
            "model",
            "horizon_step",
            "levels",
            "days",
            "nodes",
            "node_levels",
            "actual",
            "bands",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "model": {"type": "string"},
            "horizon_step": {"type": "integer", "minimum": 1},
            "levels": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            },
            "days": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "nodes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "node_levels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "actual": {"type": "array", "items": _series_schema(), "minItems": 1},
            "bands": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "array", "items": _series_schema()},
            },
        },
    },
}


def validate(name: str, payload: dict[str, Any]) -> None:
    """Check a payload against its schema before it reaches the page.

    Args:
        name: The file name, which is the schema's name.
        payload: The payload.

    Raises:
        KeyError: There is no schema by that name.
        ValueError: The payload does not satisfy it, with the failing path named.
    """
    import jsonschema

    schema = SCHEMAS[name]
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.absolute_path) or "(root)"
        raise ValueError(f"{name} fails its schema at {where}: {first.message}")


def write(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    """Validate a payload and write it.

    Args:
        directory: Where the page's data lives.
        name: The file name, which is the schema's name.
        payload: The payload.

    Returns:
        The file written.

    Raises:
        ValueError: The payload does not satisfy its schema, or holds a value JSON cannot
            carry.
    """
    validate(name, payload)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path
