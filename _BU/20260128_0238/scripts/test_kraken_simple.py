"""
Simple test for KrakenOHLCV with proper rate limiting.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Enable logging with INFO level only (reduce noise)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s - %(message)s"
)

from src.ohlcv import KrakenOHLCV


def main():
    """Test Kraken OHLCV with rate limiting."""
    print("=" * 80)
    print("Testing KrakenOHLCV with Rate Limiting")
    print("=" * 80)
    print("\nNote: Kraken API rate limit is ~15 requests/second")
    print("Using 0.5s delay between requests to stay well under limit\n")
    
    kraken = KrakenOHLCV()
    
    # Test 1: Short period (no pagination needed)
    print("\n1️⃣  Short period (7 days, ~7 bars, 1 request):")
    result = kraken.get("XBTUSD", "7d", interval_min=1440)
    df = result["XBTUSD"]
    if not df.empty:
        print(f"   ✓ {len(df)} bars fetched")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    else:
        print(f"   ✗ No data")
    
    # Test 2: Medium period (requires pagination)
    print("\n2️⃣  Medium period (6 months, ~180 bars, ~1 request):")
    result = kraken.get("XBTUSD", "6m", interval_min=1440)
    df = result["XBTUSD"]
    if not df.empty:
        print(f"   ✓ {len(df)} bars fetched")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    else:
        print(f"   ✗ No data")
    
    # Test 3: Multiple symbols (with delays)
    print("\n3️⃣  Multiple symbols (7 days each):")
    result = kraken.get(["XBTUSD", "ETHUSD"], "7d", interval_min=1440)
    for sym, df in sorted(result.items()):
        if not df.empty:
            print(f"   ✓ {sym}: {len(df)} bars")
        else:
            print(f"   ✗ {sym}: No data")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()