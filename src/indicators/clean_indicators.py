"""
Standardizes indicator data from IndicatorFetcher over a fixed period.

Uses a continuous daily time index, fills gaps with ffill/bfill, and aligns
all requested symbols to the same date range (present - period to present).
"""

from __future__ import annotations

import re

import pandas as pd

from .get_indicators import IndicatorFetcher


def _parse_period(period: str) -> int:
    """Parse period string (e.g. '3y', '2y', '7y') into number of years."""
    s = str(period).strip().lower()
    m = re.match(r"^(\d+)\s*y(?:ears?)?$", s)
    if m:
        years = int(m.group(1))
        # FIX: Add bounds validation
        if years <= 0 or years > 100:
            raise ValueError(f"Period years must be between 1 and 100, got {years}")
        return years
    raise ValueError(f"Invalid period {period!r}; expected forms like '3y', '2y', '7y'.")


def _datetime_column(df: pd.DataFrame) -> str | None:
    """Return the name of the datetime column, or None if index is the datetime."""
    for c in df.columns:
        if c.lower() in ("date", "datetime", "time", "timestamp"):
            return c
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # index is the datetime
    return "date" if "date" in df.columns else None


def _is_time_series(df: pd.DataFrame, date_col: str | None) -> bool:
    """True if the DataFrame has multiple rows and a usable datetime dimension."""
    if df.empty or len(df) < 2:
        return False
    if date_col is not None:
        return date_col in df.columns
    return isinstance(df.index, pd.DatetimeIndex)


def _to_datetime_index(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    """Ensure df has a tz-naive DatetimeIndex; use date_col if provided, else index."""
    if date_col and date_col in df.columns:
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        # FIX: Strip timezone to ensure tz-naive
        if out[date_col].dt.tz is not None:
            out[date_col] = out[date_col].dt.tz_localize(None)
        out = out.dropna(subset=[date_col])
        out = out.set_index(date_col)
        return out
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        # FIX: Strip timezone from index
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        return out
    return df.copy()


class CleanIndicators:
    """
    Standardizes indicator and index data over a fixed period.

    Uses IndicatorFetcher to fetch data, then:
    - Restricts to date range [present - period, present].
    - Builds a continuous daily time index over that range.
    - Reindexes each series to that index, adding rows where there are gaps.
    - Fills new/missing values with ffill then bfill on all data columns.

    Period is specified as a string such as "3y", "2y", or "7y".
    """

    def __init__(self, fetcher: IndicatorFetcher | None = None):
        """
        Parameters
        ----------
        fetcher : IndicatorFetcher or None
            Fetcher used to load raw data. If None, a default IndicatorFetcher
            is created.
        """
        if fetcher is None:
            fetcher = IndicatorFetcher()
        self._fetcher = fetcher

    def get_clean_indicators(
        self,
        symbols: list[str],
        period: str,
        *,
        ticker: str | None = None,
        freq: str = "D",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch and standardize indicators for the given symbols and period.

        Parameters
        ----------
        symbols : list of str
            Indicator/symbol names (e.g. "UNRATE", "sp500", "vix", "rsi").
            Same names as accepted by IndicatorFetcher.get_indicators.
        period : str
            Lookback length: "3y", "2y", "7y", etc. (number + "y").
            Date range is [present - period, present].
        ticker : str or None
            Optional stock ticker for yfinance stock-level indicators.
        freq : str
            Target frequency for the continuous time index. Default "D" (calendar
            day). Use "B" for business days.

        Returns
        -------
        dict[str, pd.DataFrame]
            One DataFrame per symbol. Each has a continuous DatetimeIndex
            from (present - period) to present, with missing rows added and
            data columns filled via ffill then bfill. Point-in-time indicators
            (e.g. one-row P/E) are broadcast to the full index as constant
            series.
        """
        years = _parse_period(period)
        end = pd.Timestamp.now().normalize()
        start = end - pd.DateOffset(years=years)

        raw = self._fetcher.get_indicators(
            symbols,
            start_date=start,
            end_date=end,
            ticker=ticker,
        )

        # FIX: Ensure full_index is tz-naive
        full_index = pd.date_range(start=start, end=end, freq=freq, tz=None)

        out: dict[str, pd.DataFrame] = {}
        for name, df in raw.items():
            if df is None or df.empty:
                continue
            cleaned = self._standardize_one(name, df, full_index, start, end)
            if cleaned is not None and not cleaned.empty:
                out[name] = cleaned

        return out

    def _standardize_one(
        self,
        name: str,
        df: pd.DataFrame,
        full_index: pd.DatetimeIndex,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Reindex one DataFrame to full_index and ffill/bfill; or broadcast if point-in-time."""
        date_col = _datetime_column(df)

        # Point-in-time: one or two columns (e.g. metric/value), single row
        if not _is_time_series(df, date_col):
            return self._broadcast_point(name, df, full_index)

        ts = _to_datetime_index(df, date_col)
        if ts.empty:
            return None

        # Drop any leftover non-datetime rows where index is NaT
        ts = ts.loc[ts.index.notna()]

        if ts.empty:
            return None

        # Reindex to full range (adds rows where index was missing)
        reindexed = ts.reindex(full_index)

        # ffill then bfill all data columns
        filled = reindexed.ffill().bfill()

        # Restore index name for clarity
        filled.index.name = "date"
        return filled

    def _broadcast_point(self, name: str, df: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Turn a one-row (point-in-time) DataFrame into a constant series over full_index."""
        # FIX: Preserve the actual column name (e.g. "value"), not the symbol name
        if "value" in df.columns:
            col_name = "value"
            val = df["value"].iloc[0]
        else:
            # Use first numeric column or first column
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    col_name = c
                    val = df[c].iloc[0]
                    break
            else:
                col_name = df.columns[0]
                val = df.iloc[0, 0]
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = float("nan")
        out = pd.DataFrame({col_name: v}, index=full_index)
        out.index.name = "date"
        return out
