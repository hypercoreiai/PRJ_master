"""
Download OHLCV candlestick data from Kraken REST API.

Uses symbol(s) (pair) and period; date range is always [current - period, current].
Kraken returns up to 720 bars per request; this class paginates to cover the
full period when needed.

Handles both single symbol and list of symbols; always returns dict.
"""

from __future__ import annotations

import time
from typing import Any

import logging
import pandas as pd
import requests

from ._period import date_range_for_period

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

# Kraken OHLC response array columns (index order)
OHLC_COLUMNS = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]

# Interval in minutes: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
DEFAULT_INTERVAL = 60  # 1 hour

# Kraken max bars per request
KRAKEN_MAX_BARS = 720

# Rate limiting (seconds)
BASE_DELAY = 0.5  # Base delay between requests (Kraken allows ~15 req/sec)
RATE_LIMIT_DELAY = 5.0  # Delay when rate limited


class KrakenOHLCV:
    """
    Downloads OHLCV from Kraken REST API (GET /0/public/OHLC).

    Date range is always [current - period, current]. Uses pagination when
    the requested range exceeds 720 bars (Kraken's per-request limit).
    
    Implements rate limiting with exponential backoff for "Too many requests" errors.
    
    Handles both single symbol and list of symbols; always returns dict.
    """

    def __init__(self, base_url: str = KRAKEN_OHLC_URL):
        self._base_url = base_url.rstrip("/")

    def get(
        self,
        symbol: str | list[str],
        period: str,
        *,
        interval_min: int = DEFAULT_INTERVAL,
    ) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV for the given pair(s) and period.

        Parameters
        ----------
        symbol : str or list[str]
            Kraken pair or list of pairs (e.g. "XBTUSD", ["XBTUSD", "ETHUSD"]).
        period : str
            Lookback length: "3y", "2y", "7y", "6m", "30d", etc. (number + unit).
            Date range is [current - period, current].
        interval_min : int
            Bar interval in minutes. Allowed: 1, 5, 15, 30, 60, 240, 1440,
            10080, 21600. Default 60 (1 hour).

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with symbol as key and DataFrame as value.
            Columns: open, high, low, close, vwap, volume, count.
            Index is DatetimeIndex named "date" (tz-naive).
            Empty DataFrame for any symbol that fails to fetch.

        Examples
        --------
        >>> kraken = KrakenOHLCV()
        >>> result = kraken.get("XBTUSD", "1y", interval_min=1440)
        {'XBTUSD': DataFrame(...)}
        
        >>> result = kraken.get(["XBTUSD", "ETHUSD"], "6m", interval_min=1440)
        {'XBTUSD': DataFrame(...), 'ETHUSD': DataFrame(...)}
        """
        # Normalize to list
        symbols = [symbol] if isinstance(symbol, str) else symbol
        symbols = [s.strip().upper() for s in symbols]
        
        # Validate interval
        allowed = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        if interval_min not in allowed:
            raise ValueError(f"interval_min must be one of {allowed}, got {interval_min}")
        
        result = {}
        for sym in symbols:
            result[sym] = self._fetch_one(sym, period, interval_min)
            # Add delay between symbols
            if sym != symbols[-1]:
                time.sleep(1.0)
        
        return result

    def _fetch_one(
        self,
        symbol: str,
        period: str,
        interval_min: int,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a single pair with pagination support.
        
        Parameters
        ----------
        symbol : str
            Single Kraken pair
        period : str
            Lookback period string
        interval_min : int
            Bar interval in minutes
        
        Returns
        -------
        pd.DataFrame
            OHLCV data or empty DataFrame if fetch fails
        """
        start_ts, end_ts = date_range_for_period(period)
        start_sec = int(start_ts.timestamp())
        end_sec = int(end_ts.timestamp())

        pair = symbol.strip().upper()
        
        # Calculate expected bars for this period
        # Kraken returns bars at interval_min granularity
        interval_sec = interval_min * 60
        expected_bars = (end_sec - start_sec) // interval_sec
        
        logging.info(
            "Fetching %s: period=%s, interval=%d min, expected ~%d bars",
            pair, period, interval_min, expected_bars
        )
        
        rows: list[list[Any]] = []
        since = start_sec
        request_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        try:
            while since < end_sec:
                request_count += 1
                
                # Exponential backoff for rate limiting
                if consecutive_errors > 0:
                    backoff = RATE_LIMIT_DELAY * (2 ** (consecutive_errors - 1))
                    backoff = min(backoff, 30)  # Cap at 30 seconds
                    logging.warning(
                        "%s: Rate limited, backing off %.1f seconds (attempt %d)",
                        pair, backoff, consecutive_errors
                    )
                    time.sleep(backoff)
                
                resp = self._request(pair, interval_min, since)
                
                if not resp:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logging.warning(
                            "%s: Max consecutive errors reached (%d), stopping",
                            pair, consecutive_errors
                        )
                        break
                    # Continue to next iteration to retry with backoff
                    continue
                
                # Reset error counter on successful response
                consecutive_errors = 0
                
                # Response keys under "result" are pair name (list of bars) or "last" (timestamp)
                result = resp.get("result", {})
                page_bars: list[list[Any]] = []
                
                for key, val in result.items():
                    if key == "last" or not isinstance(val, list):
                        continue
                    page_bars = val
                    break
                
                if not page_bars:
                    logging.info("%s: No bars returned at since=%d", pair, since)
                    break
                
                # Extend rows with new bars
                rows.extend(page_bars)
                last_bar_time = page_bars[-1][0]
                
                logging.debug(
                    "%s: Request #%d: got %d bars, last time=%d, need until %d",
                    pair, request_count, len(page_bars), last_bar_time, end_sec
                )
                
                # Check if we've reached the end
                if last_bar_time >= end_sec:
                    logging.info("%s: Reached end time, stopping pagination", pair)
                    break
                
                # Next page: since = last bar time
                since = last_bar_time
                
                # Rate limit between requests (Kraken: ~15 req/sec, use 0.5s = ~2 req/sec)
                time.sleep(BASE_DELAY)
            
            if not rows:
                logging.warning("%s: No data retrieved", pair)
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=OHLC_COLUMNS)
            df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.drop(columns=["time"])
            df = df.set_index("date")
            df.index.name = "date"
            
            # Convert to numeric types
            for c in ["open", "high", "low", "close", "vwap", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
            
            # Make index tz-naive
            if getattr(df.index, "tz", None) is not None:
                try:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                except Exception:
                    try:
                        df.index = df.index.tz_localize(None)
                    except Exception:
                        pass
            
            # Clip to requested range and sort
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            df = df.sort_index()
            
            logging.info(
                "%s: Successfully fetched %d bars (%d requests)",
                pair, len(df), request_count
            )
            
            return df
        
        except Exception as exc:
            logging.warning("Kraken OHLCV fetch failed for %s: %s", pair, exc)
            return pd.DataFrame()

    def _request(
        self,
        pair: str,
        interval: int,
        since: int
    ) -> dict[str, Any] | None:
        """
        Make a single request to Kraken OHLC endpoint.
        
        Parameters
        ----------
        pair : str
            Kraken pair
        interval : int
            Interval in minutes
        since : int
            Unix timestamp to fetch from
        
        Returns
        -------
        dict or None
            Parsed JSON response or None on error
        """
        try:
            r = requests.get(
                self._base_url,
                params={"pair": pair, "interval": interval, "since": since},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            
            if data.get("error"):
                error_msg = data.get("error")
                if error_msg:
                    # Check if it's a rate limiting error
                    error_str = str(error_msg).lower()
                    if "too many requests" in error_str or "eapi:rate limit exceeded" in error_str:
                        logging.debug(
                            "Kraken rate limit for %s: %s", pair, error_msg
                        )
                    else:
                        logging.warning("Kraken API error for %s: %s", pair, error_msg)
                return None
            
            return data
        
        except requests.exceptions.RequestException as exc:
            logging.debug("Kraken API request failed: %s", exc)
            return None
        except Exception as exc:
            logging.debug("Kraken API parsing failed: %s", exc)
            return None
