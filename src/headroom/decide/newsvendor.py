"""Staffing as a newsvendor problem, and what each forecast's staffing actually cost.

## The rule

A planner staffs each dispatch area for each day ahead. Staffing ``u`` units against
demand ``d`` costs ``cost_over`` for every unit not needed and ``cost_under`` for every unit
needed and missing. The staffing that minimises expected cost under a forecast
distribution is the **critical-ratio quantile**: staff to the demand quantile at

    q* = cost_under / (cost_under + cost_over)

and convert demand to units with the stated demand per unit, rounding up because a unit
is indivisible and a planner short of a fraction of a unit is short of a unit.
:func:`critical_ratio` is that formula and `tests/test_decide.py` checks, by brute force on
fixtures, that no other staffing level has lower expected cost.

## The oracle, and the realised cost

The oracle knew each day's demand and staffed exactly the units it needed,
``ceil(demand / demand_per_unit)``, so its cost is zero by construction. A method's
**realised cost** is what its staffing cost against that, day by day and area by area,
summed. It is the forecast priced as a decision: a method with a better CRPS that staffs
worse is caught here.

## The inputs are a table

Demand per unit and both costs are read from `inputs/decision.toml` and nowhere else. The
values shipped there are illustrative and say so; the arithmetic is the result, not the
numbers.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

#: Where the decision inputs live.
INPUTS = Path("inputs/decision.toml")


@dataclass(frozen=True, slots=True)
class DecisionInputs:
    """The replaceable inputs to the staffing decision.

    Attributes:
        demand_per_unit: Incidents one staffed unit can serve in a day.
        cost_over: Cost of a unit-day staffed and not needed.
        cost_under: Cost of a unit-day needed and not staffed.
        service_levels: Fixed service levels reported beside the critical ratio.
    """

    demand_per_unit: float
    cost_over: float
    cost_under: float
    service_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse inputs the arithmetic cannot use.

        Raises:
            ValueError: A ratio or cost is not positive, or a service level is outside
                ``(0, 1)``.
        """
        for name, value in (
            ("demand_per_unit", self.demand_per_unit),
            ("cost_over", self.cost_over),
            ("cost_under", self.cost_under),
        ):
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for level in self.service_levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"service levels must be in (0, 1), got {level}")


def load_inputs(path: Path = INPUTS) -> DecisionInputs:
    """Read the decision inputs from their table.

    Args:
        path: The TOML file.

    Returns:
        The inputs, validated.

    Raises:
        KeyError: A required input is missing from the file.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return DecisionInputs(
        demand_per_unit=float(raw["demand_per_unit"]),
        cost_over=float(raw["cost_over"]),
        cost_under=float(raw["cost_under"]),
        service_levels=tuple(float(level) for level in raw["service_levels"]),
    )


def critical_ratio(cost_over: float, cost_under: float) -> float:
    """Return the service level that minimises expected staffing cost.

    Args:
        cost_over: Cost of a unit staffed and not needed.
        cost_under: Cost of a unit needed and not staffed.

    Returns:
        ``cost_under / (cost_under + cost_over)``.
    """
    return cost_under / (cost_under + cost_over)


def demand_at(
    quantiles: npt.NDArray[np.float64],
    levels: npt.NDArray[np.float64],
    service_level: float,
) -> npt.NDArray[np.float64]:
    """Read the demand at a service level off a quantile forecast.

    Linear between grid levels. On the 199-level scoring grid every level this project
    reports is a grid point, so no interpolation happens in practice.

    Args:
        quantiles: Forecasts, shape ``(..., n_levels)``, sorted along the last axis.
        levels: The quantile grid.
        service_level: The level to read.

    Returns:
        Demand at that level, shape ``(...)``.

    Raises:
        ValueError: The level is outside the grid, which would mean inventing a tail.
    """
    if not levels[0] <= service_level <= levels[-1]:
        raise ValueError(
            f"service level {service_level} is outside the grid {levels[0]} to {levels[-1]}"
        )
    upper = int(np.searchsorted(levels, service_level))
    if math.isclose(levels[min(upper, levels.size - 1)], service_level, abs_tol=1e-9):
        return np.asarray(quantiles[..., min(upper, levels.size - 1)], dtype=np.float64)
    lower = upper - 1
    weight = (service_level - levels[lower]) / (levels[upper] - levels[lower])
    blended = (1.0 - weight) * quantiles[..., lower] + weight * quantiles[..., upper]
    return np.asarray(blended, dtype=np.float64)


def units_for(
    demand: npt.NDArray[np.float64], demand_per_unit: float
) -> npt.NDArray[np.float64]:
    """Convert demand to whole staffed units, rounding up.

    Args:
        demand: Demand in incidents.
        demand_per_unit: Incidents one unit can serve.

    Returns:
        Units, as whole numbers in a float array.
    """
    return np.ceil(np.maximum(demand, 0.0) / demand_per_unit - 1e-9)


def realised_cost(
    staffed: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    inputs: DecisionInputs,
) -> npt.NDArray[np.float64]:
    """Price a staffing decision against the oracle that knew demand.

    Args:
        staffed: Units staffed, any shape.
        actual: Demand that happened, same shape.
        inputs: Demand per unit and the two costs.

    Returns:
        Cost per element, same shape. Zero wherever staffing matched the oracle's.
    """
    needed = units_for(actual, inputs.demand_per_unit)
    over = np.maximum(staffed - needed, 0.0)
    under = np.maximum(needed - staffed, 0.0)
    return np.asarray(inputs.cost_over * over + inputs.cost_under * under, dtype=np.float64)
