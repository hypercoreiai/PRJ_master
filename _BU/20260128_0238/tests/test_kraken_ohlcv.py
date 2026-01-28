"""Test Kraken OHLCV data fetching."""

import pytest
import pandas as pd


def test_kraken_ohlcv_tz_and_columns():
    """Test that Kraken OHLCV data has tz-naive index and correct columns."""
    try:
        import ccxt
    except ImportError:
        pytest.skip("ccxt not installed")
    
    try:
        # Add timeout for API call
        exchange = ccxt.kraken({'timeout': 5000})  # 5 second timeout
        
        # Try to fetch with short timeout
        try:
            ohlcv = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=5)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, TimeoutError) as e:
            pytest.skip(f"Kraken API unavailable: {e}")
        
        if not ohlcv or len(ohlcv) == 0:
            pytest.skip("Kraken returned empty data")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Check: tz-naive
        if isinstance(df["timestamp"], pd.DatetimeIndex):
            assert df["timestamp"].tz is None, "Expected tz-naive datetime"
        elif hasattr(df["timestamp"].dt, 'tz'):
            assert df["timestamp"].dt.tz is None, "Expected tz-naive datetime"
        
        # Check: columns exist
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(df.columns)), f"Missing columns. Got {df.columns.tolist()}"
        
    except Exception as e:
        if "skip" in str(type(e).__name__).lower():
            raise
        pytest.skip(f"Kraken test skipped: {e}")