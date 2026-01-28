"""Test IndicatorFetcher technical indicators."""

import pandas as pd
from src.indicators.get_indicators import IndicatorFetcher


def test_tech_rsi_tz_naive():
    # Create tz-aware history
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-02", tz="UTC"),
        pd.Timestamp("2020-01-03", tz="UTC")
    ])
    hist = pd.DataFrame({"Close": [1.0, 2.0, 1.5]}, index=idx)
    out = IndicatorFetcher._tech_rsi(hist, "rsi")
    
    # Check output is not empty
    assert not out.empty, "RSI output should not be empty"
    
    # Check index is tz-naive
    assert out["date"].dt.tz is None, "RSI date column should be tz-naive"
    
    # Check columns exist
    assert "date" in out.columns
    assert "rsi" in out.columns
