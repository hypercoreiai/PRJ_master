"""Shared period parsing and date range for OHLCV downloaders."""

from __future__ import annotations

import re

import pandas as pd


def parse_period(period: str) -> int:
    """Parse period string (e.g. '3y', '2y', '7y') into number of years."""
    s = str(period).strip().lower()
    m = re.match(r"^(\d+)\s*y(?:ears?)?$", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid period {period!r}; expected forms like '3y', '2y', '7y'.")


def date_range_for_period(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) for current - period to current."""
    years = parse_period(period)
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=years)
    return start, end
