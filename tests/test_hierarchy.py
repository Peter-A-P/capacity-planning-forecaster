"""The summing matrix and coherence.

PLAN.md section 4 names this first among the tests that matter: if S does not reproduce
every aggregate from its leaves, every reconciliation number downstream is meaningless.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from headroom.hierarchy.build import Panel, build
from headroom.hierarchy.spec import Hierarchy, from_paths, node_id

PATHS = [("BROOKLYN", "K5"), ("BROOKLYN", "K1"), ("BRONX", "B2"), ("QUEENS", "Q1")]


def test_s_reproduces_every_aggregate_from_the_leaves():
    h = from_paths(PATHS)
    rng = np.random.default_rng(0)
    leaves = rng.uniform(0, 100, size=h.n_leaves)
    values = h.aggregate(leaves)

    assert values[h.index("NYC")] == pytest.approx(leaves.sum())
    assert values[h.index(node_id("BROOKLYN"))] == pytest.approx(
        leaves[h.leaves.index(node_id("BROOKLYN", "K1"))]
        + leaves[h.leaves.index(node_id("BROOKLYN", "K5"))]
    )
    assert values[h.index(node_id("BRONX"))] == pytest.approx(
        leaves[h.leaves.index(node_id("BRONX", "B2"))]
    )
    for leaf in h.leaves:
        assert values[h.index(leaf)] == pytest.approx(leaves[h.leaves.index(leaf)])


def test_aggregates_of_leaves_are_coherent_and_a_perturbation_is_not():
    h = from_paths(PATHS)
    rng = np.random.default_rng(1)
    values = h.aggregate(rng.uniform(0, 100, size=(h.n_leaves, 12)))

    assert h.is_coherent(values)
    assert h.coherence_error(values) < 1e-12

    broken = values.copy()
    broken[h.index(node_id("BROOKLYN")), 3] += 0.5
    assert not h.is_coherent(broken)
    assert h.coherence_error(broken) == pytest.approx(0.5)


def test_node_order_does_not_depend_on_the_order_the_paths_arrive_in():
    a = from_paths(PATHS)
    b = from_paths(list(reversed(PATHS)))
    assert a.nodes == b.nodes
    assert a.leaves == b.leaves
    assert np.array_equal(a.s_matrix, b.s_matrix)


def test_leaves_are_the_tail_of_nodes_so_the_leaf_rows_are_addressable():
    h = from_paths(PATHS)
    assert h.nodes[-h.n_leaves :] == h.leaves
    assert h.levels[: len(h.nodes) - h.n_leaves] == ("city", "borough", "borough", "borough")
    assert tuple(h.rows_at("borough")) == (1, 2, 3)


def test_an_area_under_two_boroughs_is_refused_rather_than_silently_reassigned():
    with pytest.raises(ValueError, match="both"):
        from_paths([("BROOKLYN", "K5"), ("QUEENS", "K5")])


def test_an_empty_hierarchy_is_refused():
    with pytest.raises(ValueError, match="at least one leaf"):
        from_paths([])


def test_a_hierarchy_with_a_mismatched_matrix_is_refused():
    h = from_paths(PATHS)
    with pytest.raises(ValueError, match="S is"):
        Hierarchy(h.nodes, h.leaves, h.levels, h.s_matrix[:, :1])


def _counts(rows: list[tuple[date, str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"day": pl.Date, "borough": pl.String, "area": pl.String, "n": pl.UInt32},
        orient="row",
    )


def test_build_fills_a_quiet_day_with_zero_and_keeps_the_panel_coherent():
    days = pl.date_range(date(2020, 1, 1), date(2020, 4, 9), interval="1d", eager=True)
    quiet = days[40]
    rows = [(d, "BRONX", "B1", 10) for d in days]
    rows += [(d, "BRONX", "B2", 4) for d in days if d != quiet]
    panel = build(_counts(rows))

    assert panel.days == tuple(days)
    assert panel.dropped_areas == ()
    b2 = panel.series(node_id("BRONX", "B2"))
    assert b2[40] == 0.0
    assert b2[39] == 4.0
    assert b2[41] == 4.0
    assert panel.series("NYC")[40] == 10.0
    assert panel.series("NYC")[39] == 14.0
    assert panel.hierarchy.is_coherent(panel.values)


def test_a_quiet_day_is_kept_but_a_late_opening_is_dropped():
    """The two look alike in a count of active days, and are not the same thing."""
    days = pl.date_range(date(2020, 1, 1), date(2020, 4, 9), interval="1d", eager=True)
    rows = [(d, "BRONX", "B1", 100) for d in days]
    rows += [(d, "BRONX", "QUIET", 3) for d in days[::2]]  # every other day, all period
    rows += [(d, "BRONX", "LATE", 10) for d in days[-5:]]  # opened five days before the end
    panel = build(_counts(rows))

    assert panel.dropped_areas == ("LATE",)
    assert node_id("BRONX", "QUIET") in panel.hierarchy.leaves
    assert panel.series(node_id("BRONX", "QUIET")).tolist()[:4] == [3.0, 0.0, 3.0, 0.0]


def test_a_stray_code_that_spans_the_window_is_still_dropped_on_volume():
    """Span alone would keep it: it appears in the first week and the last."""
    days = pl.date_range(date(2020, 1, 1), date(2020, 4, 9), interval="1d", eager=True)
    rows = [(d, "BRONX", "B1", 100) for d in days]
    rows += [(days[1], "BRONX", "STRAY", 1), (days[-2], "BRONX", "STRAY", 1)]
    panel = build(_counts(rows))

    assert panel.dropped_areas == ("STRAY",)
    assert panel.hierarchy.leaves == (node_id("BRONX", "B1"),)


def test_build_refuses_a_window_where_every_area_is_too_quiet():
    days = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), interval="1d", eager=True)
    rows = [(d, "BRONX", "B1", 1) for d in days[::30]]
    with pytest.raises(ValueError, match="incidents a day"):
        build(_counts(rows))


def test_build_drops_a_thinly_covered_area_and_reports_its_share():
    days = pl.date_range(date(2020, 1, 1), date(2020, 4, 9), interval="1d", eager=True)
    rows = [(d, "BRONX", "B1", 100) for d in days]
    rows += [(d, "BRONX", "LATE", 10) for d in days[-5:]]
    panel = build(_counts(rows))

    assert panel.dropped_areas == ("LATE",)
    assert panel.dropped_share == pytest.approx(50 / (100 * 100 + 50))
    assert panel.hierarchy.leaves == (node_id("BRONX", "B1"),)
    # The city is the sum of the retained leaves, which is the whole point of reporting
    # the dropped share: it is the gap between this root and every incident.
    assert panel.series("NYC").sum() == pytest.approx(100 * 100)


def test_build_refuses_a_window_nothing_spans():
    days = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), interval="1d", eager=True)
    rows = [(d, "BRONX", "B1", 1) for d in days[:10]]
    with pytest.raises(ValueError, match="no area both spans"):
        build(_counts(rows), start=date(2020, 1, 1), end=date(2020, 12, 31))


def test_a_panel_whose_values_do_not_match_its_hierarchy_is_refused():
    h = from_paths(PATHS)
    with pytest.raises(ValueError, match="values are"):
        Panel(h, (date(2020, 1, 1),), np.zeros((h.n_nodes, 2)), (), 0.0)
