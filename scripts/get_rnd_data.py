"""
Download random data from multiple sources using the correct fetchers:
- Kraken: 5 random crypto symbols (1 day of data) via KrakenOHLCV
- YFinance: 5 random stock symbols (3 years) via YFinanceOHLCV
- Indicators: 5 random indicators via IndicatorFetcher (FRED, BEA, yfinance)

Data is saved to data/raw/(symbol)_(Date).csv
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ohlcv import KrakenOHLCV, YFinanceOHLCV
from src.indicators import IndicatorFetcher


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


def select_random_symbols(symbols: list[str], count: int = 5) -> list[str]:
    """Select random symbols from list, up to count."""
    if len(symbols) <= count:
        return symbols
    return random.sample(symbols, count)


def download_kraken_data(symbols: list[str], period: str = "1y") -> None:
    """Download Kraken crypto OHLCV data."""
    print("\n" + "=" * 70)
    print("Downloading Kraken (Crypto) Data")
    print("=" * 70)
    
    kraken = KrakenOHLCV()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        try:
            print(f"\n📥 Downloading {symbol}...")
            df = kraken.get(symbol, period=period, interval_min=1440)  # 1440 min = 1 day
            
            if df is None or df.empty:
                print(f"  ✗ No data returned for {symbol}")
                continue
            
            # Save to CSV
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{symbol.replace('/', '_')}_{date_str}.csv"
            filepath = raw_dir / filename
            df.to_csv(filepath)
            print(f"  ✓ Saved {len(df)} rows to {filename}")
            
        except Exception as e:
            print(f"  ✗ Error downloading {symbol}: {e}")


def download_yfinance_data(symbols: list[str], period: str = "3y") -> None:
    """Download yfinance stock OHLCV data."""
    print("\n" + "=" * 70)
    print("Downloading YFinance (Stocks) Data")
    print("=" * 70)
    
    yfinance = YFinanceOHLCV()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        try:
            print(f"\n📥 Downloading {symbol}...")
            df = yfinance.get(symbol, period=period)
            
            if df is None or df.empty:
                print(f"  ✗ No data returned for {symbol}")
                continue
            
            # Save to CSV
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{symbol}_{date_str}.csv"
            filepath = raw_dir / filename
            df.to_csv(filepath)
            print(f"  ✓ Saved {len(df)} rows to {filename}")
            
        except Exception as e:
            print(f"  ✗ Error downloading {symbol}: {e}")


def download_indicators_data(symbols: list[str]) -> None:
    """Download indicators data using IndicatorFetcher (FRED, BEA, yfinance)."""
    print("\n" + "=" * 70)
    print("Downloading Indicators Data (FRED, BEA, yfinance)")
    print("=" * 70)
    
    fetcher = IndicatorFetcher()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Set date range for 30 years back
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=30)
    
    for symbol in symbols:
        try:
            print(f"\n📥 Downloading {symbol}...")
            
            # Use IndicatorFetcher to get data from correct source (FRED, BEA, or yfinance)
            results = fetcher.get_indicators(
                [symbol],
                start_date=start_date,
                end_date=end_date
            )
            
            if symbol not in results or results[symbol] is None or results[symbol].empty:
                print(f"  ✗ No data returned for {symbol}")
                continue
            
            df = results[symbol]
            
            # Save to CSV
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{symbol}_{date_str}.csv"
            filepath = raw_dir / filename
            df.to_csv(filepath, index=False)
            print(f"  ✓ Saved {len(df)} rows to {filename}")
            
        except Exception as e:
            print(f"  ✗ Error downloading {symbol}: {e}")


def main():
    """Main script logic."""
    print("=" * 70)
    print("Random Data Download from Multiple Sources")
    print("=" * 70)
    
    # Define paths
    kraken_path = project_root / "src" / "symbols" / "kraken"
    indicators_path = project_root / "src" / "symbols" / "indicators"
    yfinance_path = project_root / "src" / "symbols" / "yfinance"
    
    # Read symbols from each source
    print("\n📋 Reading symbol files...")
    kraken_symbols = read_symbols(kraken_path)
    indicators_symbols = read_symbols(indicators_path)
    yfinance_symbols = read_symbols(yfinance_path)
    
    # Select random symbols
    print("\n🎲 Selecting random symbols...")
    random_kraken = select_random_symbols(kraken_symbols, 5)
    random_indicators = select_random_symbols(indicators_symbols, 5)
    random_yfinance = select_random_symbols(yfinance_symbols, 5)
    
    print(f"✓ Selected {len(random_kraken)} Kraken symbols: {random_kraken}")
    print(f"✓ Selected {len(random_indicators)} Indicators: {random_indicators}")
    print(f"✓ Selected {len(random_yfinance)} YFinance symbols: {random_yfinance}")
    
    # Download data using appropriate sources
    if random_kraken:
        download_kraken_data(random_kraken, period="1y")
    
    if random_yfinance:
        download_yfinance_data(random_yfinance, period="3y")
    
    if random_indicators:
        download_indicators_data(random_indicators)
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    raw_dir = project_root / "data" / "raw"
    if raw_dir.exists():
        files = list(raw_dir.glob("*.csv"))
        print(f"✓ Total files downloaded: {len(files)}")
        print(f"✓ Data saved to: {raw_dir}")
        for f in sorted(files)[-10:]:  # Show last 10 files
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("⚠️  No data directory found")


if __name__ == "__main__":
    main()