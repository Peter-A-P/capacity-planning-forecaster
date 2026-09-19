"""The newsvendor staffing rule and the realised cost against an oracle.

The test PLAN.md names is the second one: on fixtures, the staffing level the cost ratio
picks has lower expected cost than every other staffing level, found by brute force rather
than by trusting the formula.
"""

from dataclasses import replace

import numpy as np
import pytest

from headroom.decide.newsvendor import (
    INPUTS,
    DecisionInputs,
    critical_ratio,
    demand_at,
    load_inputs,
    realised_cost,
    units_for,
)
from headroom.score import levels as lv

INPUTS_4_TO_1 = DecisionInputs(
    demand_per_unit=10.0,
    cost_over=1.0,
    cost_under=4.0,
    hours_per_unit_day=24.0,
    service_levels=(0.8, 0.9, 0.95),
)


def test_the_critical_ratio_is_the_share_of_cost_in_understaffing():
    assert critical_ratio(cost_over=1.0, cost_under=4.0) == pytest.approx(0.8)
    assert critical_ratio(cost_over=3.0, cost_under=1.0) == pytest.approx(0.25)


@pytest.mark.parametrize(("cost_over", "cost_under"), [(1.0, 4.0), (1.0, 1.0), (3.0, 1.0)])
def test_staffing_to_the_critical_ratio_beats_every_other_staffing_level(cost_over, cost_under):
    rng = np.random.default_rng(0)
    demand = rng.gamma(shape=9.0, scale=11.0, size=20_001)  # right-skewed, like arrivals
    inputs = replace(INPUTS_4_TO_1, cost_over=cost_over, cost_under=cost_under)
    q = critical_ratio(cost_over, cost_under)
    chosen = units_for(
        np.array(np.quantile(demand, q, method="inverted_cdf")), inputs.demand_per_unit
    )

    expected = {
        units: realised_cost(np.full(demand.size, float(units)), demand, inputs).mean()
        for units in range(0, 40)
    }
    best = min(expected, key=lambda units: expected[units])
    assert float(chosen) == best


def test_the_oracle_costs_nothing():
    actual = np.array([3.0, 10.0, 10.5, 47.0, 0.0])
    oracle = units_for(actual, 10.0)
    np.testing.assert_array_equal(realised_cost(oracle, actual, INPUTS_4_TO_1), 0.0)


def test_each_unit_short_costs_the_under_cost_and_each_spare_the_over_cost():
    actual = np.array([35.0, 35.0])  # needs 4 units
    staffed = np.array([2.0, 7.0])
    np.testing.assert_array_equal(realised_cost(staffed, actual, INPUTS_4_TO_1), [8.0, 3.0])


def test_units_round_up_but_exact_multiples_do_not_gain_a_unit():
    np.testing.assert_array_equal(
        units_for(np.array([20.0, 20.01, 0.0, -3.0]), 10.0), [2.0, 3.0, 0.0, 0.0]
    )


def test_demand_is_read_at_a_grid_level_exactly_and_between_levels_linearly():
    levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.array([[10.0, 20.0, 40.0]])
    assert demand_at(quantiles, levels, 0.5)[0] == 20.0
    assert demand_at(quantiles, levels, 0.7)[0] == pytest.approx(30.0)
    with pytest.raises(ValueError, match="outside the grid"):
        demand_at(quantiles, levels, 0.95)


def test_every_reported_service_level_is_a_point_of_the_scoring_grid():
    inputs = load_inputs(INPUTS)
    for level in (*inputs.service_levels, critical_ratio(inputs.cost_over, inputs.cost_under)):
        assert np.isclose(lv.SCORING, level, atol=1e-9).any()


def test_the_shipped_inputs_table_loads_and_bad_inputs_are_refused():
    inputs = load_inputs(INPUTS)
    assert inputs.demand_per_unit > 0
    assert inputs.cost_under > 0
    assert inputs.hours_per_unit_day > 0
    with pytest.raises(ValueError, match="must be positive"):
        replace(inputs, cost_over=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        replace(inputs, hours_per_unit_day=0.0)
    with pytest.raises(ValueError, match="service levels"):
        replace(inputs, service_levels=(1.0,))
