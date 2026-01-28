"""
Download OHLCV candlestick data from Kraken REST API.

Uses symbol (pair) and period; date range is always [current - period, current].
Kraken returns up to 720 bars per request; this class paginates to cover the
full period when needed.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from ._period import date_range_for_period

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

# Kraken OHLC response array columns (index order)
OHLC_COLUMNS = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]

# Interval in minutes: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
DEFAULT_INTERVAL = 60  # 1 hour


class KrakenOHLCV:
    """
    Downloads OHLCV from Kraken REST API (GET /0/public/OHLC).

    Date range is always [current - period, current]. Uses pagination when
    the requested range exceeds 720 bars (Kraken's per-request limit).
    """

    def __init__(self, base_url: str = KRAKEN_OHLC_URL):
        self._base_url = base_url.rstrip("/")

    def get(
        self,
        symbol: str,
        period: str,
        *,
        interval_min: int = DEFAULT_INTERVAL,
    ) -> pd.DataFrame:
        """
        Download OHLCV for the given pair and period.

        Parameters
        ----------
        symbol : str
            Kraken pair (e.g. "XBTUSD", "XXBTZUSD", "ETHUSD").
        period : str
            Lookback length: "3y", "2y", "7y", etc. (number + "y").
            Date range is [current - period, current].
        interval_min : int
            Bar interval in minutes. Allowed: 1, 5, 15, 30, 60, 240, 1440,
            10080, 21600. Default 60 (1 hour).

        Returns
        -------
        pd.DataFrame
            Columns: time (datetime index), open, high, low, close, vwap,
            volume, count. Index name "date". Empty DataFrame on error.
        """
        start_ts, end_ts = date_range_for_period(period)
        start_sec = int(start_ts.timestamp())
        end_sec = int(end_ts.timestamp())

        allowed = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        if interval_min not in allowed:
            raise ValueError(f"interval_min must be one of {allowed}, got {interval_min}")

        pair = symbol.strip().upper()
        since = start_sec
        rows: list[list[Any]] = []

        while since < end_sec:
            resp = self._request(pair, interval_min, since)
            if not resp:
                break
            # Response keys under "result" are pair name (list of bars) or "last" (timestamp)
            result = resp.get("result", {})
            page_bars: list[list[Any]] = []
            for key, val in result.items():
                if key == "last" or not isinstance(val, list):
                    continue
                page_bars = val
                break
            if not page_bars:
                break
            rows.extend(page_bars)
            # Next page: since = last bar time + 1 from this response
            last_ts = page_bars[-1][0]
            if last_ts >= end_sec:
                break
            since = last_ts + 1
            time.sleep(0.2)  # rate limit

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=OHLC_COLUMNS)
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.drop(columns=["time"])
        df = df.set_index("date")
        df.index.name = "date"
        for c in ["open", "high", "low", "close", "vwap", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
        # Naive index for consistent comparison with start_ts/end_ts
        df.index = df.index.tz_localize(None)
        # Clip to requested range
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        return df.sort_index()

    def _request(self, pair: str, interval: int, since: int) -> dict[str, Any] | None:
        try:
            r = requests.get(
                self._base_url,
                params={"pair": pair, "interval": interval, "since": since},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                return None
            return data
        except Exception:
            return None
