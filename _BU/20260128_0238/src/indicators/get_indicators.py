"""
Financial indicators fetcher from yfinance, FRED, and BEA.

Supports stock valuation, fundamentals, technicals, market data (yfinance),
economic indicators (FRED), and BEA NIPA data (PCE inflation, Real GDP growth,
Personal/Disposable Income). FRED_API_KEY and BEA_API_KEY in .env for those sources.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Optional

import logging
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# FRED series mapping: user-facing name -> FRED series_id
# ---------------------------------------------------------------------------
FRED_INDICATORS: dict[str, str] = {
    # Interest rates
    "T10Y2Y": "T10Y2Y",                    # 10-Year/2-Year Treasury Spread
    "TB3MS": "TB3MS",                      # 3-Month Treasury Bill Rate
    "MORTGAGE30US": "MORTGAGE30US",        # 30-Year Conventional Mortgage Rate
    # Inflation & prices
    "CPIAUCSL": "CPIAUCSL",                # Consumer Price Index
    "PCE": "PCE",                          # Personal Consumption Expenditures
    # Economic activity
    "GDPC1": "GDPC1",                      # Real GDP (level)
    "A191RL1Q225SBEA": "A191RL1Q225SBEA",  # GDP Growth Rate (Real GDP % change)
    "INDPRO": "INDPRO",                    # Industrial Production Index
    "RSXFS": "RSXFS",                      # Advance Real Retail and Food Services Sales
    # Labor market
    "UNRATE": "UNRATE",                    # Civilian Unemployment Rate
    "PAYEMS": "PAYEMS",                    # Total Nonfarm Payrolls
    # Financial & sentiment
    "BAMLC0A4CBBB": "BAMLC0A4CBBB",        # ICE BofA BBB US Corporate Index (proxy for corp spreads)
    "TDSP": "TDSP",                        # Household Debt Service Payments % of Disposable Income
    "UMCSENT": "UMCSENT",                  # U. of Michigan Consumer Sentiment
}

# Aliases for easier names
FRED_ALIASES: dict[str, str] = {
    "treasury_spread_10y2y": "T10Y2Y",
    "treasury_3m": "TB3MS",
    "mortgage_30y": "MORTGAGE30US",
    "cpi": "CPIAUCSL",
    "pce": "PCE",
    "gdp_growth": "A191RL1Q225SBEA",
    "gdp": "GDPC1",
    "industrial_production": "INDPRO",
    "retail_sales": "RSXFS",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "corporate_bond_spread": "BAMLC0A4CBBB",
    "household_debt_service": "TDSP",
    "consumer_sentiment": "UMCSENT",
}

# ---------------------------------------------------------------------------
# BEA (Bureau of Economic Analysis) via beaapi: (datasetname, TableName, Frequency, line_selector)
# line_selector: int = LineNumber, or str = substring in LineDescription
# ---------------------------------------------------------------------------
BEA_INDICATORS: dict[str, tuple[str, str, str, int | str]] = {
    "BEA_GDP_GROWTH": ("NIPA", "T10101", "Q", 1),           # Real GDP percent change (Table 1.1.1)
    "BEA_PCE_PRICE": ("NIPA", "T20304", "Q", 1),            # PCE price index (Table 2.3.4)
    "BEA_PERSONAL_INCOME": ("NIPA", "T20101", "Q", 1),      # Personal income (Table 2.1)
    "BEA_DISPOSABLE_INCOME": ("NIPA", "T20101", "Q", 2),    # Disposable personal income (Table 2.1)
}

BEA_ALIASES: dict[str, str] = {
    "bea_gdp_growth": "BEA_GDP_GROWTH",
    "bea_pce_inflation": "BEA_PCE_PRICE",
    "bea_pce_price": "BEA_PCE_PRICE",
    "bea_personal_income": "BEA_PERSONAL_INCOME",
    "bea_disposable_income": "BEA_DISPOSABLE_INCOME",
}

# yfinance: stock-level indicators need a ticker; index/market use fixed tickers
YF_INDEX_TICKERS: dict[str, str] = {
    "sp500": "^GSPC",
    "vix": "^VIX",
    "s&p 500": "^GSPC",
    "s&p500": "^GSPC",
}

# Human-readable or alternate names -> canonical key for yfinance stock indicators
YF_STOCK_ALIASES: dict[str, str] = {
    "p/e": "pe",
    "pe": "pe",
    "p_e": "pe",
    "price_to_earnings": "pe",
    "forward p/e": "forward_pe",
    "forward_pe": "forward_pe",
    "forward_p_e": "forward_pe",
    "p/b": "pb",
    "pb": "pb",
    "price_to_book": "pb",
    "ev/ebitda": "ev_ebitda",
    "ev_ebitda": "ev_ebitda",
    "dividend_yield": "dividend_yield",
    "dividend yield": "dividend_yield",
    "peg": "peg_ratio",
    "peg_ratio": "peg_ratio",
    "peg ratio": "peg_ratio",
    "eps": "eps",
    "earnings per share": "eps",
    "revenue_growth": "revenue_growth",
    "revenue growth": "revenue_growth",
    "free_cash_flow": "free_cash_flow",
    "free cash flow": "free_cash_flow",
    "fcf": "free_cash_flow",
    "debt_to_equity": "debt_to_equity",
    "debt to equity": "debt_to_equity",
    "roe": "roe",
    "return on equity": "roe",
    "return_on_equity": "roe",
    "net_profit_margin": "net_profit_margin",
    "profit margin": "net_profit_margin",
    "sma": "sma",
    "moving average": "sma",
    "simple moving average": "sma",
    "ema": "ema",
    "exponential moving average": "ema",
    "rsi": "rsi",
    "relative strength index": "rsi",
    "macd": "macd",
    "volume": "volume",
    "bollinger": "bollinger",
    "bollinger bands": "bollinger",
    "bollinger_bands": "bollinger",
    "historical": "historical",
    "ohlcv": "ohlcv",
    "prices": "historical",
    "open high low close": "historical",
    "open high low close volume": "ohlcv",
}

# Default date range when not provided
_DEFAULT_END = datetime.now()
_DEFAULT_START = _DEFAULT_END - timedelta(days=365 * 5)


def _resolve_fred_id(name: str) -> str | None:
    key = name.strip().upper()
    if key in FRED_INDICATORS:
        return FRED_INDICATORS[key]
    low = name.strip().lower()
    return FRED_ALIASES.get(low)


def _is_fred_indicator(name: str) -> bool:
    return _resolve_fred_id(name) is not None


def _is_yf_index(name: str) -> bool:
    k = name.strip().lower()
    return k in (x.lower() for x in YF_INDEX_TICKERS)


def _yf_index_ticker(name: str) -> str | None:
    k = name.strip().lower()
    for alias, ticker in YF_INDEX_TICKERS.items():
        if alias == k:
            return ticker
    return None


def _resolve_bea_spec(name: str) -> tuple[str, str, str, int | str] | None:
    key = name.strip().upper()
    if key in BEA_INDICATORS:
        return BEA_INDICATORS[key]
    low = name.strip().lower()
    alias_key = BEA_ALIASES.get(low)
    return BEA_INDICATORS.get(alias_key) if alias_key else None


class IndicatorFetcher:
    """
    Fetches financial and economic indicators from yfinance, FRED, and BEA.

    - yfinance: stock valuation, fundamentals, technicals, and market/index data.
    - FRED: interest rates, inflation, GDP, labor, and sentiment (FRED_API_KEY in .env).
    - BEA: PCE inflation/price, Real GDP growth, Personal/Disposable Income via beaapi
      (BEA_API_KEY in .env).

    Accepts a list of indicator names and returns a dict of name -> DataFrame.
    """

    # Indicator names that require a ticker when using yfinance (stock-level)
    YF_STOCK_INDICATORS = frozenset({
        "pe", "forward_pe", "pb", "ev_ebitda", "dividend_yield", "peg_ratio",
        "eps", "revenue_growth", "free_cash_flow", "debt_to_equity", "roe",
        "net_profit_margin",
        "sma", "ema", "rsi", "macd", "volume", "bollinger",
        "historical", "ohlcv", "prices",
    })

    # Names that map to index tickers (no user ticker needed)
    YF_INDEX_NAMES = frozenset({"sp500", "vix", "s&p 500", "s&p500"})

    def __init__(
        self,
        fred_api_key: str | None = None,
        bea_api_key: str | None = None,
    ):
        self._fred_key = fred_api_key or os.getenv("FRED_API_KEY")
        self._fred: Any = None
        if self._fred_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self._fred_key)
            except ImportError:
                self._fred = None
        self._bea_key = bea_api_key or os.getenv("BEA_API_KEY") or os.getenv("BEA_KEY")

    def get_indicators(
        self,
        indicator_names: list[str],
        *,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        ticker: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch requested indicators and return a dict of name -> DataFrame.

        Parameters
        ----------
        indicator_names : list of str
            Names such as "T10Y2Y", "UNRATE", "sp500", "vix", "pe", "rsi", etc.
        start_date, end_date : optional
            Date range for time series. Defaults to 5 years back until now.
        ticker : optional
            Stock symbol for yfinance stock-level indicators (e.g. "AAPL").
            Required when requesting P/E, EPS, RSI, etc. Ignored for FRED and
            index symbols (sp500, vix).

        Returns
        -------
        dict[str, pd.DataFrame]
            One DataFrame per requested indicator; key = normalized indicator name.
        """
        start = start_date or _DEFAULT_START
        end = end_date or _DEFAULT_END
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        out: dict[str, pd.DataFrame] = {}

        for name in indicator_names:
            n = name.strip()
            # 1) FRED
            fid = _resolve_fred_id(n)
            if fid is not None:
                df = self._fetch_fred(fid, start, end)
                if df is not None and not df.empty:
                    out[n] = df
                continue

            # 2) BEA (beaapi)
            bea_spec = _resolve_bea_spec(n)
            if bea_spec is not None:
                df = self._fetch_bea(bea_spec, start, end)
                if df is not None and not df.empty:
                    out[n] = df
                continue

            # 3) yfinance index (sp500, vix)
            yf_ticker = _yf_index_ticker(n)
            if yf_ticker is not None:
                df = self._fetch_yf_series(yf_ticker, start, end, label=n)
                if df is not None and not df.empty:
                    out[n] = df
                continue

            # 4) yfinance stock-level (pe, rsi, historical, etc.)
            key = n.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            key = YF_STOCK_ALIASES.get(key, key)
            if key in self.YF_STOCK_INDICATORS or key in {"ohlcv", "prices", "historical"}:
                sym = ticker
                if not sym:
                    continue  # skip if no ticker provided
                df = self._fetch_yf_stock_indicator(sym, key, start, end)
                if df is not None and not df.empty:
                    out[n] = df
                continue

        return out

    def _fetch_fred(self, series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
        if self._fred is None:
            return None
        try:
            start_str = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
            end_str = end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)
            s = self._fred.get_series(
                series_id,
                observation_start=start_str,
                observation_end=end_str,
            )
            if s is None or (hasattr(s, "empty") and s.empty):
                return None
            df = pd.DataFrame({"value": s})
            df.index.name = "date"
            return df.reset_index()
        except Exception as exc:
            logging.exception("FRED fetch failed: %s", exc)
            return None

    def _fetch_bea(
        self,
        spec: tuple[str, str, str, int | str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        if not self._bea_key:
            return None
        datasetname, table_name, freq, line_sel = spec
        try:
            import beaapi
        except ImportError:
            return None
        years = range(start.year, end.year + 1)
        year_str = ",".join(str(y) for y in years)
        try:
            tbl = beaapi.get_data(
                self._bea_key,
                datasetname=datasetname,
                TableName=table_name,
                Frequency=freq,
                Year=year_str,
            )
        except Exception as exc:
            logging.exception("BEA API call failed: %s", exc)
            return None
        if tbl is None or tbl.empty or "TimePeriod" not in tbl.columns or "DataValue" not in tbl.columns:
            return None
        if isinstance(line_sel, int):
            tbl = tbl.loc[tbl["LineNumber"].astype(int) == line_sel]
        else:
            mask = tbl["LineDescription"].astype(str).str.contains(str(line_sel), case=False, na=False)
            tbl = tbl.loc[mask]
        if tbl.empty:
            return None
        tbl = tbl[["TimePeriod", "DataValue"]].copy()
        tbl["DataValue"] = pd.to_numeric(tbl["DataValue"], errors="coerce")
        tbl = tbl.dropna(subset=["DataValue"])
        # Parse TimePeriod: "2015Q1" -> period end, "2015" -> year end
        def _parse_period(s: str) -> pd.Timestamp | None:
            s = str(s).strip().upper()
            if len(s) == 4 and s.isdigit():
                return pd.Timestamp(year=int(s), month=12, day=31)
            if len(s) == 6 and "Q" in s:
                parts = s.split("Q")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    y, q = int(parts[0]), int(parts[1])
                    if 1 <= q <= 4:
                        month = q * 3
                        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(0)
            return None
        tbl["date"] = tbl["TimePeriod"].map(_parse_period)
        tbl = tbl.dropna(subset=["date"])
        tbl = tbl.rename(columns={"DataValue": "value"})[["date", "value"]]
        tbl = tbl.sort_values("date").reset_index(drop=True)
        tbl = tbl.loc[(tbl["date"] >= start) & (tbl["date"] <= end)]
        return tbl if not tbl.empty else None

    def _fetch_yf_series(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        label: str,
    ) -> pd.DataFrame | None:
        try:
            import yfinance as yf
            obj = yf.Ticker(symbol)
            hist = obj.history(start=start, end=end)
            if hist is None or hist.empty:
                return None
            hist = hist.copy()
            hist.index.name = "date"
            # standardize columns and make index tz-naive
            hist = hist.rename(columns={c: c.lower() for c in hist.columns})
            if getattr(hist.index, "tz", None) is not None:
                try:
                    hist.index = hist.index.tz_convert("UTC").tz_localize(None)
                except Exception:
                    try:
                        hist.index = hist.index.tz_localize(None)
                    except Exception:
                        pass
            return hist.reset_index()
        except Exception as exc:
            logging.exception("yfinance series fetch failed: %s", exc)
            return None

    def _fetch_yf_stock_indicator(
        self,
        ticker: str,
        key: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info
            hist = t.history(start=start, end=end)

            if key in {"historical", "ohlcv", "prices"}:
                if hist is not None and not hist.empty:
                    hist = hist.copy()
                    hist.index.name = "date"
                    hist = hist.rename(columns={c: c.lower() for c in hist.columns})
                    if getattr(hist.index, "tz", None) is not None:
                        try:
                            hist.index = hist.index.tz_convert("UTC").tz_localize(None)
                        except Exception:
                            try:
                                hist.index = hist.index.tz_localize(None)
                            except Exception:
                                pass
                    return hist.reset_index()
                return None

            # Single-point valuation/fundamental metrics -> one-row DataFrame
            val_map = {
                "pe": "trailingPE",
                "forward_pe": "forwardPE",
                "pb": "priceToBook",
                "ev_ebitda": "enterpriseToEbitda",
                "dividend_yield": "dividendYield",
                "peg_ratio": "pegRatio",
                "eps": "trailingEps",
                "debt_to_equity": "debtToEquity",
                "roe": "returnOnEquity",
                "net_profit_margin": "profitMargins",
            }
            if key in val_map:
                attr = val_map[key]
                v = info.get(attr)
                if v is None:
                    return None
                return pd.DataFrame([{"metric": key, "value": v}])

            if key == "revenue_growth":
                v = info.get("revenueGrowth")
                if v is None:
                    return None
                return pd.DataFrame([{"metric": "revenue_growth", "value": v}])

            if key == "free_cash_flow":
                v = info.get("freeCashflow")
                if v is None:
                    return None
                return pd.DataFrame([{"metric": "free_cash_flow", "value": v}])

            # Technicals: time series
            if hist is None or hist.empty:
                return None

            if key == "sma":
                return self._tech_sma(hist, 20, key)
            if key == "ema":
                return self._tech_ema(hist, 20, key)
            if key == "rsi":
                return self._tech_rsi(hist, key)
            if key == "macd":
                return self._tech_macd(hist, key)
            if key == "volume":
                df = hist[["Volume"]].copy()
                df.columns = [key]
                df.index.name = "date"
                return df.reset_index()
            if key == "bollinger":
                return self._tech_bollinger(hist, key)

            return None
        except Exception as exc:
            logging.exception("yfinance stock indicator fetch failed: %s", exc)
            return None

    @staticmethod
    def _tech_sma(hist: pd.DataFrame, period: int, name: str) -> pd.DataFrame:
        c = "Close" if "Close" in hist.columns else hist.columns[0]
        s = hist[c].rolling(period, min_periods=1).mean()
        idx = s.index
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
        out = pd.DataFrame({"date": idx, name: s.values})
        return out

    @staticmethod
    def _tech_ema(hist: pd.DataFrame, period: int, name: str) -> pd.DataFrame:
        c = "Close" if "Close" in hist.columns else hist.columns[0]
        s = hist[c].ewm(span=period, adjust=False).mean()
        idx = s.index
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
        out = pd.DataFrame({"date": idx, name: s.values})
        return out

    @staticmethod
    def _tech_rsi(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Compute RSI and return with tz-naive date column."""
        try:
            import talib
            use_talib = True
        except ImportError:
            use_talib = False
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # FIX: Strip timezone to ensure tz-naive datetime
        dates = df.index
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        
        close = df["Close"]
        
        if use_talib:
            rsi = talib.RSI(close.values, timeperiod=14)
        else:
            # Simple RSI calculation without talib
            rsi = pd.Series(index=close.index, dtype=float)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.values
        
        # Create output with tz-naive date column
        out = pd.DataFrame({
            "date": dates,
            "rsi": rsi
        })
        
        # Don't drop NaN rows - keep all rows
        return out

    @staticmethod
    def _tech_macd(hist: pd.DataFrame, name: str) -> pd.DataFrame:
        c = "Close" if "Close" in hist.columns else hist.columns[0]
        ema12 = hist[c].ewm(span=12, adjust=False).mean()
        ema26 = hist[c].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        idx = hist.index
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
        out = pd.DataFrame({
            "date": idx,
            "macd": macd_line.values,
            "macd_signal": signal.values,
            "macd_hist": (macd_line - signal).values,
        })
        return out.dropna(how="all", subset=["macd", "macd_signal", "macd_hist"])

    @staticmethod
    def _tech_bollinger(hist: pd.DataFrame, name: str) -> pd.DataFrame:
        c = "Close" if "Close" in hist.columns else hist.columns[0]
        period, std_dev = 20, 2.0
        mid = hist[c].rolling(period, min_periods=1).mean()
        std = hist[c].rolling(period, min_periods=1).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        idx = hist.index
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
        out = pd.DataFrame({
            "date": idx,
            "bollinger_lower": lower.values,
            "bollinger_mid": mid.values,
            "bollinger_upper": upper.values,
        })
        return out
