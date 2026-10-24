"""Smoke test: the Streamlit app renders without exceptions."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_runs_and_shows_prediction():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any(m.label == "Churn probability" for m in at.metric)


def test_app_handles_no_internet_customer():
    at = AppTest.from_file(APP, default_timeout=60).run()
    internet = next(s for s in at.sidebar.selectbox if s.label == "Internet service")
    internet.select("No").run()
    assert not at.exception
