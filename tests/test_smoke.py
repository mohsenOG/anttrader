"""Integration checks for the command-line pipeline and Streamlit scaffold."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from anttrader.pipeline import DEFAULT_CONFIG, main


def pipeline_config(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["historical_csv"] = "synthetic.csv"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_pipeline_reports_missing_dataset(tmp_path, capsys):
    main(pipeline_config(tmp_path))
    output = capsys.readouterr().out
    assert "historical CSV missing" in output
    assert "Dukascopy XAU/USD H1 bid" in output
    assert "README.md" in output


def test_pipeline_runs_data_to_grid(tmp_path, capsys):
    path = pipeline_config(tmp_path)
    close = np.arange(100, 357)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=257, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    ).to_csv(tmp_path / "synthetic.csv", index=False)
    main(path)
    output = capsys.readouterr().out
    assert "257 validated completed candles" in output
    assert "Historical grids generated: 1" in output
    assert "Latest decision timestamp: 2024-01-11 17:00:00+00:00" in output
    assert "Grid shape: (16, 16); values: [1]" in output


def test_pipeline_reports_invalid_config(tmp_path):
    path = pipeline_config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["lookback_returns"] = 255
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="lookback_returns must equal"):
        main(path)


def test_dashboard_renders_status():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert not app.exception
    assert app.title[0].value == "AntTrader"
    assert [tab.label for tab in app.tabs] == [
        "Ant Simulation",
        "Gold & Signals",
    ]
    assert "signal research tool" in app.warning[0].value
