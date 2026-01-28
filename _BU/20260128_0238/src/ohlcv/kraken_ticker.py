"""
Download ticker (current price/market) data from Kraken REST API.

Uses symbol (pair) and period for interface consistency; date range is
[current - period, current]. Ticker data is a point-in-time snapshot
as of the request time.
"""

from __future__ import annotations

from typing import Any

import logging
import pandas as pd
import requests

from ._period import date_range_for_period

KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"


class KrakenTicker:
    """
    Downloads ticker information from Kraken REST API (GET /0/public/Ticker).

    Returns current bid/ask/last/volume etc. for the given pair. Period is
    accepted for a consistent API; the returned snapshot is as of the
    request time (end of the period).
    """

    def __init__(self, base_url: str = KRAKEN_TICKER_URL):
        self._base_url = base_url.rstrip("/")

    def get(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Download ticker data for the given pair.

        Parameters
        ----------
        symbol : str
            Kraken pair (e.g. "XBTUSD", "XXBTZUSD", "ETHUSD").
        period : str
            Lookback length: "3y", "2y", "7y", etc. Used only to fix the
            "as of" date: snapshot is taken at current time (end of period).

        Returns
        -------
        pd.DataFrame
            One row of ticker fields (ask, bid, last, volume, etc.) with
            columns flattened from Kraken’s nested format. Includes a "date"
            column set to the request time (normalized). Empty DataFrame on
            error.
        """
        _, end = date_range_for_period(period)
        raw = self._request(symbol.strip().upper())
        if not raw:
            return pd.DataFrame()
        row = self._flatten_ticker(raw, end)
        if not row:
            return pd.DataFrame()
        return pd.DataFrame([row])

    def _request(self, pair: str) -> dict[str, Any] | None:
        try:
            r = requests.get(self._base_url, params={"pair": pair}, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                return None
            return data
        except Exception as exc:
            logging.exception("Kraken Ticker request failed: %s", exc)
            return None

    def _flatten_ticker(self, data: dict[str, Any], as_of: pd.Timestamp) -> dict[str, Any]:
        result = data.get("result") or {}
        # Kraken returns pair name as key; use first pair’s data
        for key, val in result.items():
            if key == "last" or not isinstance(val, dict):
                continue
            out: dict[str, Any] = {"date": as_of, "pair": key}
            for k, v in val.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    out[f"{k}"] = v[0]
                    out[f"{k}_lot"] = v[1]
                else:
                    out[k] = v
            return out
        return {}
