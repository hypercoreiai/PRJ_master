"""
Download and clean random indicators using CleanIndicators.

- Reads random indicators from src/symbols/indicators (only working ones)
- Fetches max available data (30 years for FRED, available for BEA)
- Standardizes to continuous daily time index
- Fills gaps with ffill/bfill
- Saves to data/clean/(symbol)_(Date).csv
"""

import sys
from pathlib import Path
from datetime import datetime
import random
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.clean_indicators import CleanIndicators


# Indicators known to work with IndicatorFetcher
WORKING_INDICATORS = [
    # FRED indicators (economic data) - ALL WORKING ✅
    "T10Y2Y",           # 10-year / 2-year yield spread
    "UNRATE",           # Unemployment rate
    "UMCSENT",          # Consumer sentiment
    "CPIAUCSL",         # CPI
    "INDPRO",           # Industrial production
    "PAYEMS",           # Nonfarm payroll
    "HOUST",            # Housing starts
    
    # BEA indicators (national accounts) - PARTIALLY WORKING ✅
    "BEA_GDP_GROWTH",   # Real GDP growth
    
    # Stock indices via yfinance - USE get_rnd_data.py instead
    # "^GSPC",          # These don't work with CleanIndicators
    # "^VIX",           # Use YFinanceOHLCV in get_rnd_data.py
]


def read_symbols(filepath: Path) -> list[str]:
    """Read symbols from a file, one per line, stripping whitespace and comments."""
    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        return []
    
    with open(filepath, "r") as f:
        symbols = [
            line.strip() 
            for line in f 
            if line.strip() and not line.strip().startswith("#")
        ]
    
    print(f"✓ Read {len(symbols)} symbols from {filepath.name}")
    return symbols


def get_working_indicators(all_symbols: list[str]) -> list[str]:
    """Filter to only indicators known to work."""
    working = [s for s in all_symbols if s in WORKING_INDICATORS]
    print(f"✓ Found {len(working)} working indicators out of {len(all_symbols)}")
    return working


def select_random_symbols(symbols: list[str], count: int = 5) -> list[str]:
    """Select random symbols from list, up to count."""
    if len(symbols) <= count:
        return symbols
    return random.sample(symbols, count)


def download_and_clean_indicators(
    symbols: list[str],
    period: str = "3y",
    freq: str = "D"
) -> dict[str, pd.DataFrame]:
    """
    Download and clean indicators using CleanIndicators.
    
    Parameters
    ----------
    symbols : list[str]
        Indicator symbols to fetch
    period : str
        Lookback period (e.g., "3y", "5y", "30y")
    freq : str
        Frequency for continuous index ("D" for daily, "B" for business)
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames, keyed by symbol
    """
    print("\n" + "=" * 70)
    print(f"Downloading and Cleaning Indicators ({period})")
    print("=" * 70)
    
    clean = CleanIndicators()
    raw_dir = project_root / "data" / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Fetch each indicator individually for better error handling
    print(f"\n📥 Fetching {len(symbols)} indicators individually...\n")
    for symbol in symbols:
        try:
            print(f"  📥 {symbol:20s}...", end=" ")
            individual_result = clean.get_clean_indicators(
                [symbol],
                period=period,
                freq=freq
            )
            
            if individual_result and symbol in individual_result:
                df = individual_result[symbol]
                results[symbol] = df
                print(f"✓ ({len(df):6d} rows)")
            else:
                print(f"✗ No data")
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}")
    
    # Save results to CSV
    if results:
        print(f"\n✓ Successfully cleaned {len(results)}/{len(symbols)} indicators")
        print("\n📝 Saving to CSV files...")
        for symbol, df in results.items():
            try:
                filename = f"{symbol}_{date_str}.csv"
                filepath = raw_dir / filename
                df.to_csv(filepath)
                print(f"  ✓ {symbol}: {len(df)} rows → {filename}")
            except Exception as e:
                print(f"  ✗ Error saving {symbol}: {e}")
    else:
        print("✗ No indicators were successfully fetched!")
    
    return results


def print_summary(results: dict[str, pd.DataFrame]) -> None:
    """Print summary of downloaded and cleaned data."""
    print("\n" + "=" * 70)
    print("Clean Indicators Summary")
    print("=" * 70)
    
    clean_dir = project_root / "data" / "clean"
    if clean_dir.exists():
        files = list(clean_dir.glob("*.csv"))
        print(f"✓ Total files in directory: {len(files)}")
        print(f"✓ Data saved to: {clean_dir}\n")
        
        # Print statistics for each indicator
        if results:
            print("Successfully Cleaned Indicators:")
            print("-" * 70)
            for symbol, df in sorted(results.items()):
                if df.empty:
                    print(f"  {symbol:20s} - EMPTY")
                else:
                    date_range = f"{df.index.min().date()} to {df.index.max().date()}"
                    cols = ", ".join(df.columns.tolist())
                    print(f"  {symbol:20s} - {len(df):6d} rows | {date_range} | {cols}")
        else:
            print("⚠️  No indicators were successfully cleaned")
    else:
        print("⚠️  No clean data directory found")


def main():
    """Main script logic."""
    print("=" * 70)
    print("Clean Indicators Download and Standardization")
    print("=" * 70)
    
    # Define path
    indicators_path = project_root / "src" / "symbols" / "indicators"
    
    # Read all symbols
    print("\n📋 Reading indicator symbols...")
    all_symbols = read_symbols(indicators_path)
    
    # Filter to working indicators
    print("\n🔍 Filtering to working indicators...")
    working_indicators = get_working_indicators(all_symbols)
    
    # Select random symbols
    print("\n🎲 Selecting random indicators...")
    random_indicators = select_random_symbols(working_indicators, 5)
    print(f"✓ Selected {len(random_indicators)} indicators:")
    for sym in sorted(random_indicators):
        print(f"    - {sym}")
    
    # Download and clean
    results = download_and_clean_indicators(
        random_indicators,
        period="3y",   # 3 years for better data coverage
        freq="D"       # Daily frequency
    )
    
    # Print summary
    print_summary(results)
    
    # Show sample data
    if results:
        print("\n" + "=" * 70)
        print("Sample Data (first 10 rows of each indicator)")
        print("=" * 70)
        for symbol in sorted(results.keys()):
            df = results[symbol]
            print(f"\n{symbol} ({len(df)} rows, {df.index.min().date()} to {df.index.max().date()}):")
            print(df.head(10))


if __name__ == "__main__":
    main()