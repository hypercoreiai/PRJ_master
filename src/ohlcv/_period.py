"""
Parse period strings and compute date ranges for OHLCV downloads.

Supports flexible period formats: "3y", "6m", "30d", "1y", etc.
Date range is always [current - period, current].
"""

from __future__ import annotations

from datetime import datetime, timedelta
import re


def parse_period(period: str) -> tuple[int, int, int]:
    """
    Parse period string into (years, months, days).
    
    Parameters
    ----------
    period : str
        Period string like "3y", "6m", "30d", "1y6m", "2y3m15d", etc.
        Supported units: y (year), m (month), d (day).
    
    Returns
    -------
    tuple[int, int, int]
        (years, months, days)
    
    Raises
    ------
    ValueError
        If period format is invalid
    
    Examples
    --------
    >>> parse_period("3y")
    (3, 0, 0)
    
    >>> parse_period("6m")
    (0, 6, 0)
    
    >>> parse_period("30d")
    (0, 0, 30)
    
    >>> parse_period("1y6m15d")
    (1, 6, 15)
    """
    s = str(period).strip().lower()
    
    if not s:
        raise ValueError(f"Invalid period {period!r}; expected non-empty string")
    
    # Match all (number)(unit) pairs: 3y, 6m, 30d, etc.
    pattern = r"(\d+)([ymd])"
    matches = re.findall(pattern, s)
    
    if not matches:
        raise ValueError(
            f"Invalid period {period!r}; expected forms like '3y', '6m', '30d', '1y6m15d', etc."
        )
    
    years, months, days = 0, 0, 0
    seen_units = set()
    
    for value_str, unit in matches:
        value = int(value_str)
        
        if unit in seen_units:
            raise ValueError(
                f"Invalid period {period!r}; duplicate unit '{unit}'"
            )
        seen_units.add(unit)
        
        if unit == "y":
            years = value
        elif unit == "m":
            months = value
        elif unit == "d":
            days = value
    
    # Check that units are in order (y, then m, then d)
    unit_order = {"y": 0, "m": 1, "d": 2}
    unit_sequence = [unit_order[u] for _, u in matches]
    if unit_sequence != sorted(unit_sequence):
        raise ValueError(
            f"Invalid period {period!r}; units must be in order (y, m, d)"
        )
    
    return years, months, days


def date_range_for_period(period: str) -> tuple[datetime, datetime]:
    """
    Compute date range for a given period.
    
    Parameters
    ----------
    period : str
        Period string like "3y", "6m", "30d", "1y6m", etc.
    
    Returns
    -------
    tuple[datetime, datetime]
        (start_date, end_date) where end = now, start = now - period.
        Both are tz-naive UTC datetimes at 00:00:00.
    
    Examples
    --------
    >>> start, end = date_range_for_period("3y")
    >>> end.year - start.year >= 3
    True
    """
    years, months, days = parse_period(period)
    
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Subtract years and months using replace if possible
    # Handle year/month rollover manually
    start_year = end.year - years
    start_month = end.month - months
    
    # Roll back months and years as needed
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    
    # Create start date with same day as end (or end of month if end.day doesn't exist)
    try:
        start = end.replace(year=start_year, month=start_month)
    except ValueError:
        # Day doesn't exist in start month (e.g., Feb 31), use last day of month
        import calendar
        last_day = calendar.monthrange(start_year, start_month)[1]
        start = end.replace(year=start_year, month=start_month, day=last_day)
    
    # Subtract remaining days
    if days > 0:
        start = start - timedelta(days=days)
    
    return start, end
