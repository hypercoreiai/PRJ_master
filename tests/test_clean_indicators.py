"""Test CleanIndicators with a fake fetcher."""

import pandas as pd
from src.indicators.clean_indicators import CleanIndicators


class FakeFetcher:
    """Fake fetcher for testing."""

    def get_indicators(self, symbols, start_date=None, end_date=None, ticker=None):
        """Return fake data for 'pe' (point-in-time) and 'sp' (time series)."""
        out = {}
        
        # Point-in-time: single row with 'value' column
        if "pe" in symbols:
            out["pe"] = pd.DataFrame({"value": [42.0]})
        
        # Time series: multiple rows with date column and 'value' column
        if "sp" in symbols:
            dates = pd.date_range(start_date, end_date, freq="D")
            # Create time series with values that grow from 100 to 101
            values = [100.0 + (i / len(dates)) for i in range(len(dates))]
            out["sp"] = pd.DataFrame({
                "date": dates,
                "value": values
            })
        
        return out


def test_clean_indicators_broadcast_and_tz():
    fetcher = FakeFetcher()
    c = CleanIndicators(fetcher=fetcher)
    out = c.get_clean_indicators(["pe", "sp"], "1y", ticker=None, freq="D")

    # 'pe' should be broadcast to full index
    assert "pe" in out
    df_pe = out["pe"]
    assert len(df_pe) > 1
    
    # 'sp' should be present and have tz-naive index
    assert "sp" in out
    df_sp = out["sp"]
    assert df_sp.index.tz is None
    
    # values propagated - check the 'value' column
    assert "value" in df_sp.columns, f"Expected 'value' column, got {df_sp.columns.tolist()}"
    assert df_sp["value"].iloc[0] == 100.0 or df_sp["value"].iloc[-1] > 100.0
