"""
Download OHLCV price data from yfinance.

Uses symbol(s) and period; date range is always [current - period, current].
Handles both single symbol and list of symbols; always returns dict.
"""

from __future__ import annotations

import logging
import pandas as pd

from ._period import date_range_for_period


class YFinanceOHLCV:
    """
    Downloads OHLCV (Open, High, Low, Close, Volume) from yfinance.

    Date range is always [current - period, current] for the given period
    (e.g. "3y", "2y", "7y").
    
    Handles both single symbol and list of symbols; always returns dict.
    """

    def get(
        self,
        symbol: str | list[str],
        period: str
    ) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV for the given symbol(s) and period.

        Parameters
        ----------
        symbol : str or list[str]
            Ticker symbol or list of symbols (e.g. "AAPL", ["AAPL", "BTC-USD", "^GSPC"]).
        period : str
            Lookback length: "3y", "2y", "7y", etc. (number + "y").
            Date range is [current - period, current].

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with symbol as key and DataFrame as value.
            Columns: open, high, low, close, volume (and dividends, stock_splits
            if returned by yfinance). Index is DatetimeIndex named "date" (tz-naive).
            Empty DataFrame for any symbol that fails to fetch.

        Examples
        --------
        >>> yf = YFinanceOHLCV()
        >>> result = yf.get("AAPL", "3y")
        {'AAPL': DataFrame(...)}
        
        >>> result = yf.get(["AAPL", "BTC-USD"], "2y")
        {'AAPL': DataFrame(...), 'BTC-USD': DataFrame(...)}
        """
        # Normalize to list
        symbols = [symbol] if isinstance(symbol, str) else symbol
        symbols = [s.strip() for s in symbols]
        
        result = {}
        for sym in symbols:
            result[sym] = self._fetch_one(sym, period)
        
        return result

    def _fetch_one(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Fetch OHLCV for a single symbol.
        
        Parameters
        ----------
        symbol : str
            Single ticker symbol
        period : str
            Lookback period string
        
        Returns
        -------
        pd.DataFrame
            OHLCV data or empty DataFrame if fetch fails
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
            # Standardize columns to lowercase
            hist = hist.rename(columns={c: c.lower() for c in hist.columns})
            # Make index tz-naive
            if getattr(hist.index, "tz", None) is not None:
                try:
                    hist.index = hist.index.tz_convert("UTC").tz_localize(None)
                except Exception:
                    try:
                        hist.index = hist.index.tz_localize(None)
                    except Exception:
                        pass
            return hist
        except Exception as exc:
            logging.warning("yfinance OHLCV fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()
