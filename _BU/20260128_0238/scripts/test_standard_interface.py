"""
Test standard interface for OHLCV classes.
All should handle single symbol, list of symbols, and always return dict.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Enable logging to see pagination details
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)

from src.ohlcv import YFinanceOHLCV, KrakenOHLCV


def test_kraken():
    """Test KrakenOHLCV standard interface."""
    print("\n" + "=" * 80)
    print("Testing KrakenOHLCV Standard Interface")
    print("=" * 80)
    
    kraken = KrakenOHLCV()
    
    # Test 1: Single symbol (string)
    print("\n1️⃣  Single symbol (string):")
    result = kraken.get("XBTUSD", "1m", interval_min=1440)
    print(f"   Type: {type(result)}")
    print(f"   Keys: {list(result.keys())}")
    if "XBTUSD" in result:
        df = result["XBTUSD"]
        print(f"   XBTUSD shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   ✓ PASS")
    else:
        print(f"   ✗ FAIL - XBTUSD not in result")
    
    # Test 2: Multiple symbols (list)
    print("\n2️⃣  Multiple symbols (list):")
    result = kraken.get(["XBTUSD", "ETHUSD"], "7d", interval_min=1440)
    print(f"   Type: {type(result)}")
    print(f"   Keys: {sorted(result.keys())}")
    print(f"   Symbols: {len(result)}")
    all_present = all(k in result for k in ["XBTUSD", "ETHUSD"])
    if all_present:
        print(f"   ✓ PASS - All symbols present")
        for sym, df in result.items():
            if not df.empty:
                print(f"      {sym}: {df.shape[0]} rows")
            else:
                print(f"      {sym}: empty")
    else:
        print(f"   ✗ FAIL - Not all symbols present")
    
    # Test 3: Various period formats (with 720-bar limit explanation)
    print("\n3️⃣  Various period formats (720-bar Kraken limit per request):")
    print("   Note: Larger periods require pagination (multiple requests)\n")
    
    periods = [
        ("7d", 1440, "~7 bars = 1 request"),
        ("1m", 1440, "~30 bars = 1 request"),
        ("3m", 1440, "~90 bars = 1 request"),
        ("6m", 1440, "~180 bars = 1 request"),
        ("1y", 1440, "~365 bars = 1 request (near limit)"),
    ]
    
    for period_str, interval, desc in periods:
        try:
            result = kraken.get("XBTUSD", period_str, interval_min=interval)
            rows = result["XBTUSD"].shape[0] if not result["XBTUSD"].empty else 0
            print(f"   {period_str:6s} @ {interval} min → {rows:4d} rows ({desc})")
        except Exception as e:
            print(f"   {period_str:6s} @ {interval} min → ERROR: {str(e)[:40]}")


def test_yfinance():
    """Test YFinanceOHLCV standard interface."""
    print("=" * 80)
    print("Testing YFinanceOHLCV Standard Interface")
    print("=" * 80)
    
    yf = YFinanceOHLCV()
    
    # Test 1: Single symbol (string)
    print("\n1️⃣  Single symbol (string):")
    result = yf.get("AAPL", "1y")
    print(f"   Type: {type(result)}")
    print(f"   Keys: {list(result.keys())}")
    if "AAPL" in result:
        df = result["AAPL"]
        print(f"   AAPL shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   ✓ PASS")
    else:
        print(f"   ✗ FAIL - AAPL not in result")
    
    # Test 2: Multiple symbols (list)
    print("\n2️⃣  Multiple symbols (list):")
    result = yf.get(["AAPL", "MSFT", "BTC-USD"], "6m")
    print(f"   Type: {type(result)}")
    print(f"   Keys: {sorted(result.keys())}")
    print(f"   Symbols: {len(result)}")
    all_present = all(k in result for k in ["AAPL", "MSFT", "BTC-USD"])
    if all_present:
        print(f"   ✓ PASS - All symbols present")
        for sym, df in result.items():
            print(f"      {sym}: {df.shape[0]} rows")
    else:
        print(f"   ✗ FAIL - Not all symbols present")
    
    # Test 3: Various period formats
    print("\n3️⃣  Various period formats:")
    periods = ["30d", "3m", "6m", "1y", "2y"]
    for p in periods:
        try:
            result = yf.get("AAPL", p)
            rows = result["AAPL"].shape[0] if not result["AAPL"].empty else 0
            print(f"   {p:6s} → {rows:4d} rows ✓")
        except Exception as e:
            print(f"   {p:6s} → ERROR: {str(e)[:40]}")


def test_period_parsing():
    """Test period parsing with various formats."""
    print("\n" + "=" * 80)
    print("Testing Period Parsing")
    print("=" * 80)
    
    from src.ohlcv._period import parse_period
    
    test_cases = [
        ("30d", (0, 0, 30)),
        ("3m", (0, 3, 0)),
        ("6m", (0, 6, 0)),
        ("1y", (1, 0, 0)),
        ("2y", (2, 0, 0)),
        ("1y6m", (1, 6, 0)),
        ("1y6m15d", (1, 6, 15)),
        ("6m30d", (0, 6, 30)),
    ]
    
    print("\nPeriod Format Tests:")
    for period_str, expected in test_cases:
        try:
            result = parse_period(period_str)
            status = "✓" if result == expected else "✗"
            print(f"   {status} {period_str:12s} → {result}")
        except Exception as e:
            print(f"   ✗ {period_str:12s} → ERROR: {str(e)[:40]}")


if __name__ == "__main__":
    test_period_parsing()
    test_yfinance()
    test_kraken()
    
    print("\n" + "=" * 80)
    print("Standard Interface Tests Complete")
    print("=" * 80)