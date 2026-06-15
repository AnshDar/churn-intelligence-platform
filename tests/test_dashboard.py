"""Smoke test for the Streamlit dashboard.

Uses Streamlit's AppTest harness to execute the script in-process and assert it
renders without raising. Skipped when the scoring DB is empty (e.g. a fresh
checkout before the simulation has been run).
"""
from __future__ import annotations

import pytest

from src.database import fetch_all


@pytest.mark.skipif(fetch_all().empty, reason="no scoring data — run `python -m src.simulate` first")
def test_dashboard_renders_without_error():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dashboard/app.py", default_timeout=90).run()
    assert at.exception is None or len(at.exception) == 0, at.exception
    assert [t.label for t in at.tabs] == [
        "Executive Summary", "Risk Distribution", "At-Risk Customers",
        "Explainability", "Model Performance", "Business Metrics",
    ]
    assert len(at.metric) > 0
