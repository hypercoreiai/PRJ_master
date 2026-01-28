"""
Download OHLCV price data from yfinance.

Uses symbol and period; date range is always [current - period, current].
"""

from __future__ import annotations

import pandas as pd

from ._period import date_range_for_period


class YFinanceOHLCV:
    """
    Downloads OHLCV (Open, High, Low, Close, Volume) from yfinance.

    Date range is always [current - period, current] for the given period
    (e.g. "3y", "2y", "7y").
    """

    def get(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Download OHLCV for the given symbol and period.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. "AAPL", "BTC-USD", "^GSPC").
        period : str
            Lookback length: "3y", "2y", "7y", etc. (number + "y").
            Date range is [current - period, current].

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume (and Dividends, Stock Splits
            if returned by yfinance). Index is DatetimeIndex named "date".
            Empty DataFrame if fetch fails.
        """
        start, end = date_range_for_period(period)
        try:
            import yfinance as yf
            obj = yf.Ticker(symbol)
            hist = obj.history(start=start, end=end)
            if hist is None or hist.empty:
                return pd.DataFrame()
            hist = hist.copy()
            hist.index.name = "date"
            return hist
        except Exception:
            return pd.DataFrame()
