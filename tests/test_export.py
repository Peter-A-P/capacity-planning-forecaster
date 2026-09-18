"""The dashboard's payloads, its schema, and the static page that reads them.

The page is served from a CDN with no backend, so nothing about it fails loudly at run
time: a payload that has lost a field is a blank panel on someone else's machine, and a
content security policy that the page itself violates is a chart that silently does not
draw. Both of those are checked here instead.

The last group of tests reads the files in `dashboard/` as they will be published.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from headroom.report import export as ex

DASHBOARD = Path("dashboard")
DATA = DASHBOARD / "data"


def estimate(point: float) -> tuple[float, float, float]:
    return (point, point - 1.0, point + 1.0)


def dashboard_payload() -> dict[str, Any]:
    """A small payload with every section filled, shaped as the CLI shapes it."""
    days = [date(2020, 1, 5), date(2020, 1, 12), date(2020, 1, 19)]
    return ex.dashboard_payload(
        generated=date(2026, 9, 18),
        origins=ex.origins_section(days=days, step=7, horizon=14, train_window=1095),
        hierarchy=ex.hierarchy_section(
            nodes=["NYC", "NYC/A", "NYC/A/1"], levels=["city", "borough", "area"]
        ),
        coverage=ex.coverage_section(
            level="city",
            days=days,
            window_origins=2,
            shift=(date(2020, 3, 1), date(2020, 6, 1)),
            methods=[
                (
                    "split",
                    0.9,
                    np.array([np.nan, 0.88, 0.91]),
                    np.array([np.nan, 120.0, 140.0]),
                )
            ],
        ),
        reconciliation=ex.reconciliation_section(
            model="ETS",
            base_coherence_error=514.8,
            reconciled_coherence_error=5.46e-12,
            negative_share=0.0001,
            nominal=0.9,
            rows=[
                ex.reconciliation_row(
                    level="City",
                    base_crps=estimate(120.0),
                    variants=[("ETS + MinT", estimate(-8.4), estimate(0.9))],
                )
            ],
        ),
        staffing=ex.staffing_section(
            demand_per_unit=40.0,
            cost_over=1.0,
            cost_under=3.0,
            implied=0.75,
            oracle_units=90.0,
            reference="ETS",
            service_levels=[0.75, 0.9],
            methods=[
                ex.staffing_method(
                    name="ETS",
                    units=[estimate(95.0), estimate(101.0)],
                    cost=[estimate(12.0), estimate(15.0)],
                    cost_change=[None, None],
                ),
                ex.staffing_method(
                    name="Seasonal naive",
                    units=[estimate(99.0), estimate(106.0)],
                    cost=[estimate(14.0), estimate(18.0)],
                    cost_change=[estimate(2.0), estimate(3.0)],
                ),
            ],
        ),
    )


def forecast_payload() -> dict[str, Any]:
    days = [date(2020, 1, 19), date(2020, 1, 26)]
    bands = np.arange(2 * len(ex.FAN_LEVELS) * 2, dtype=float).reshape(2, len(ex.FAN_LEVELS), 2)
    return ex.forecast_payload(
        model="ETS",
        horizon_step=14,
        days=days,
        nodes=["NYC", "NYC/A"],
        node_levels=["city", "borough"],
        actual=np.array([[100.0, 110.0], [50.0, np.nan]]),
        bands=bands,
    )


def test_a_whole_dashboard_payload_satisfies_its_schema():
    ex.validate(ex.DASHBOARD, dashboard_payload())


def test_a_whole_forecast_payload_satisfies_its_schema():
    ex.validate(ex.FORECAST, forecast_payload())


def test_the_schema_catches_a_section_that_has_lost_a_field():
    payload = dashboard_payload()
    del payload["staffing"]["oracle_units"]
    with pytest.raises(ValueError, match="staffing"):
        ex.validate(ex.DASHBOARD, payload)


def test_the_schema_catches_a_value_of_the_wrong_type():
    payload = dashboard_payload()
    payload["origins"]["count"] = "911"
    with pytest.raises(ValueError, match="origins/count"):
        ex.validate(ex.DASHBOARD, payload)


def test_the_schema_catches_a_page_written_for_another_version():
    payload = dashboard_payload()
    payload["schema_version"] = ex.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        ex.validate(ex.DASHBOARD, payload)


def test_a_missing_measurement_becomes_null_and_never_nan():
    payload = ex.coverage_section(
        level="city",
        days=[date(2020, 1, 5), date(2020, 1, 12)],
        window_origins=2,
        shift=(date(2020, 3, 1), date(2020, 6, 1)),
        methods=[("split", 0.9, np.array([np.nan, 0.9]), np.array([np.nan, 12.0]))],
    )
    assert payload["series"][0]["coverage"] == [None, 0.9]
    assert "NaN" not in json.dumps(payload)


def test_an_interval_that_does_not_contain_its_point_is_refused():
    # The block bootstrap does this on a window of a few blocks, and a chart drawn from it
    # would show an interval sitting beside its own estimate (`docs/methods.md`).
    with pytest.raises(ValueError, match="does not contain"):
        ex.estimate((0.1126, 0.1171, 0.1853), 4)


def test_a_coverage_series_must_have_one_value_per_origin():
    with pytest.raises(ValueError, match="for 2 origins"):
        ex.coverage_section(
            level="city",
            days=[date(2020, 1, 5), date(2020, 1, 12)],
            window_origins=2,
            shift=(date(2020, 3, 1), date(2020, 6, 1)),
            methods=[("split", 0.9, np.array([0.9]), np.array([12.0]))],
        )


def test_a_staffing_method_must_be_priced_at_every_service_level():
    with pytest.raises(ValueError, match="service levels"):
        ex.staffing_section(
            demand_per_unit=40.0,
            cost_over=1.0,
            cost_under=3.0,
            implied=0.75,
            oracle_units=90.0,
            reference="ETS",
            service_levels=[0.75, 0.9],
            methods=[
                ex.staffing_method(
                    name="ETS",
                    units=[estimate(95.0)],
                    cost=[estimate(12.0)],
                    cost_change=[None],
                )
            ],
        )


def test_the_reference_method_has_to_be_in_the_staffing_table():
    with pytest.raises(ValueError, match="reference method"):
        ex.staffing_section(
            demand_per_unit=40.0,
            cost_over=1.0,
            cost_under=3.0,
            implied=0.75,
            oracle_units=90.0,
            reference="ETS",
            service_levels=[0.9],
            methods=[
                ex.staffing_method(
                    name="Seasonal naive",
                    units=[estimate(99.0)],
                    cost=[estimate(14.0)],
                    cost_change=[None],
                )
            ],
        )


def test_the_fan_bands_have_to_match_the_nodes_and_days_they_claim():
    with pytest.raises(ValueError, match="bands are"):
        ex.forecast_payload(
            model="ETS",
            horizon_step=14,
            days=[date(2020, 1, 19)],
            nodes=["NYC"],
            node_levels=["city"],
            actual=np.array([[100.0]]),
            bands=np.zeros((1, 2, 1)),
        )


def test_the_hierarchy_needs_a_level_for_every_node():
    with pytest.raises(ValueError, match="2 nodes and 1 levels"):
        ex.hierarchy_section(nodes=["NYC", "NYC/A"], levels=["city"])


def test_a_payload_with_no_origins_is_refused():
    with pytest.raises(ValueError, match="no origins"):
        ex.origins_section(days=[], step=7, horizon=14, train_window=None)


def test_writing_validates_first_and_leaves_nothing_behind_when_it_fails(tmp_path):
    payload = dashboard_payload()
    del payload["coverage"]
    with pytest.raises(ValueError, match="fails its schema"):
        ex.write(tmp_path, ex.DASHBOARD, payload)
    assert not (tmp_path / ex.DASHBOARD).exists()


def test_what_is_written_is_json_a_browser_can_parse(tmp_path):
    path = ex.write(tmp_path, ex.DASHBOARD, dashboard_payload())
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["origins"]["count"] == 3


# The built site ----------------------------------------------------------------------

PAGE = DASHBOARD / "index.html"
CONFIG = DASHBOARD / "staticwebapp.config.json"


@pytest.mark.skipif(not PAGE.exists(), reason="the dashboard has not been built")
def test_the_page_only_asks_for_files_that_are_published():
    html = PAGE.read_text(encoding="utf-8")
    for reference in re.findall(r'(?:src|href)="([^"]+)"', html):
        if reference.startswith("http"):
            continue
        assert (DASHBOARD / reference).exists(), f"{reference} is not in {DASHBOARD}"


@pytest.mark.skipif(not PAGE.exists(), reason="the dashboard has not been built")
def test_the_stylesheet_only_asks_for_files_that_are_published():
    # The fonts are reached from the stylesheet and not from the page, so the test above
    # cannot see them, and the content security policy allows no off-origin font anyway: a
    # missing file here is a page that silently falls back to a system font.
    css = (DASHBOARD / "style.css").read_text(encoding="utf-8")
    asked = re.findall(r"url\(([^)]+)\)", css)
    assert asked, "the stylesheet asks for no files at all, which means the fonts are gone"
    for reference in asked:
        name = reference.strip("\"'")
        assert not name.startswith("http"), f"{name} is off-origin and the policy forbids it"
        assert (DASHBOARD / name).exists(), f"{name} is not in {DASHBOARD}"


@pytest.mark.skipif(not CONFIG.exists(), reason="the dashboard has not been built")
def test_every_kind_of_file_published_has_a_declared_content_type():
    # Azure and the local server both read this table. A font served as a byte stream is one
    # more way for the page a stranger sees to differ from the page that was checked.
    declared = json.loads(CONFIG.read_text(encoding="utf-8"))["mimeTypes"]
    for path in DASHBOARD.rglob("*"):
        if path.is_file():
            assert path.suffix in declared, f"{path.name} has no content type in {CONFIG.name}"


@pytest.mark.skipif(not PAGE.exists(), reason="the dashboard has not been built")
def test_the_page_carries_nothing_its_own_policy_would_refuse():
    # A page with an inline script or a style attribute needs 'unsafe-inline' in the policy,
    # which is the whole value of having one. On project 01 a local file server sent no
    # headers at all and hid exactly this for two weeks, so it is asserted here.
    html = PAGE.read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\ssrc=)", html), "an inline script"
    assert not re.search(r"<style", html), "an inline stylesheet"
    assert not re.search(r"\sstyle=", html), "an inline style attribute"
    assert not re.search(r"\son[a-z]+=", html), "an inline event handler"


@pytest.mark.skipif(not CONFIG.exists(), reason="the dashboard has not been built")
def test_the_content_security_policy_is_the_strict_one():
    policy = json.loads(CONFIG.read_text(encoding="utf-8"))["globalHeaders"][
        "Content-Security-Policy"
    ]
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy
    assert "default-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy


@pytest.mark.skipif(not (DATA / ex.DASHBOARD).exists(), reason="nothing exported yet")
def test_the_exported_dashboard_file_still_satisfies_the_schema():
    ex.validate(ex.DASHBOARD, json.loads((DATA / ex.DASHBOARD).read_text(encoding="utf-8")))


@pytest.mark.skipif(not (DATA / ex.FORECAST).exists(), reason="nothing exported yet")
def test_the_exported_forecast_file_still_satisfies_the_schema():
    payload = json.loads((DATA / ex.FORECAST).read_text(encoding="utf-8"))
    ex.validate(ex.FORECAST, payload)
    assert len(payload["bands"]) == len(payload["nodes"])
    assert all(
        len(level) == len(payload["days"]) for node in payload["bands"] for level in node
    )
