"""
Diagnostic script to test IndicatorFetcher routing for different indicator types.
Shows which source (yfinance, FRED, BEA) each indicator uses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.get_indicators import IndicatorFetcher


def test_indicator_fetch():
    """Test fetching various indicators to see which ones work."""
    
    fetcher = IndicatorFetcher()
    
    # Test cases: indicator name -> expected source
    test_indicators = [
        ("^GSPC", "yfinance"),      # Stock index
        ("eps", "yfinance"),         # Stock fundamental
        ("pb", "yfinance"),          # Stock fundamental
        ("pe", "yfinance"),          # Stock fundamental
        ("roe", "yfinance"),         # Stock fundamental
        ("T10Y2Y", "FRED"),          # Yield curve
        ("UNRATE", "FRED"),          # Unemployment rate
        ("UMCSENT", "FRED"),         # Consumer sentiment
        ("BEA_GDP_GROWTH", "BEA"),   # BEA GDP growth
        ("BEA_DISPOSABLE_INCOME", "BEA"),  # BEA income
    ]
    
    print("=" * 80)
    print("Testing Indicator Fetcher - Source Routing")
    print("=" * 80)
    
    for symbol, expected_source in test_indicators:
        try:
            print(f"\n📥 Testing {symbol:30s} (expected: {expected_source})...")
            
            # Try to fetch with a short period
            results = fetcher.get_indicators(
                [symbol],
                start_date=None,
                end_date=None
            )
            
            if symbol in results and results[symbol] is not None and not results[symbol].empty:
                df = results[symbol]
                print(f"  ✓ SUCCESS - Got {len(df)} rows")
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    Shape: {df.shape}")
                print(f"    First row:\n{df.iloc[0] if len(df) > 0 else 'N/A'}")
            else:
                print(f"  ✗ FAILED - No data returned or empty")
                
        except Exception as e:
            print(f"  ✗ ERROR - {str(e)[:100]}")


def test_clean_indicators():
    """Test CleanIndicators directly."""
    from src.indicators.clean_indicators import CleanIndicators
    
    print("\n\n" + "=" * 80)
    print("Testing CleanIndicators with Single Indicator")
    print("=" * 80)
    
    clean = CleanIndicators()
    
    test_symbols = ["^GSPC", "UMCSENT", "roe"]
    
    for symbol in test_symbols:
        try:
            print(f"\n📥 Testing CleanIndicators with {symbol}...")
            result = clean.get_clean_indicators(
                [symbol],
                period="3y",
                freq="D"
            )
            
            if result and symbol in result:
                df = result[symbol]
                print(f"  ✓ SUCCESS - Got {len(df)} rows")
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    Date range: {df.index.min()} to {df.index.max()}")
            else:
                print(f"  ✗ FAILED - CleanIndicators returned no data")
                print(f"    Result keys: {list(result.keys()) if result else 'None'}")
                
        except Exception as e:
            print(f"  ✗ ERROR - {str(e)[:100]}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_indicator_fetch()
    test_clean_indicators()