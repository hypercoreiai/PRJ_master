import types
import sys
import pandas as pd

from ohlcv.yfinance_ohlcv import YFinanceOHLCV


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None):
        idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")])
        return pd.DataFrame({"Open": [1.0], "Close": [2.0]}, index=idx)


def test_yfinance_ohlcv_tz_and_columns(monkeypatch):
    # Inject fake yfinance module
    fake_mod = types.SimpleNamespace(Ticker=FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_mod)

    v = YFinanceOHLCV()
    df = v.get("AAPL", "1y")

    assert not df.empty
    # columns lowercased
    assert "open" in df.columns and "close" in df.columns
    # index tz-naive
    assert df.index.tz is None
