"""Smoke checks for the implemented Topic 1 entry points."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from anttrader.pipeline import main


def test_pipeline_reports_placeholder_status(capsys) -> None:
    main()
    output = capsys.readouterr().out
    assert "AntTrader" in output
    assert "Research components are not implemented yet." in output


def test_dashboard_renders_placeholder_status() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert app.title[0].value == "AntTrader"
    assert "Research components are not implemented yet." in app.info[0].value
