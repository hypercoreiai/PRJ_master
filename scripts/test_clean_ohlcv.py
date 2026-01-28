"""
Test CleanOHLCV with data from get_rnd_data.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ohlcv.clean_ohlcv import CleanOHLCV


def load_raw_ohlcv(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load OHLCV CSV files from raw data directory."""
    ohlcv_dict = {}
    
    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {data_dir}")
        return ohlcv_dict
    
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"⚠️  No CSV files found in {data_dir}")
        return ohlcv_dict
    
    print(f"📂 Found {len(csv_files)} CSV files")
    
    for filepath in sorted(csv_files):
        try:
            # Extract symbol from filename (e.g., "BTC-USD_20260128.csv" -> "BTC-USD")
            symbol = filepath.stem.rsplit("_", 1)[0]
            
            # Load CSV
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Standardize column names to uppercase
            df.columns = [c.strip().title() for c in df.columns]
            
            ohlcv_dict[symbol] = df
            print(f"  ✓ Loaded {symbol}: {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error loading {filepath.name}: {e}")
    
    return ohlcv_dict


def main():
    """Main test logic."""
    print("=" * 80)
    print("Testing CleanOHLCV")
    print("=" * 80)
    
    # Load raw OHLCV data
    print("\n📥 Loading raw OHLCV data...")
    raw_dir = project_root / "data" / "raw"
    ohlcv_dict = load_raw_ohlcv(raw_dir)
    
    if not ohlcv_dict:
        print("⚠️  No data to clean")
        return
    
    # Clean as dictionary
    print("\n🧹 Cleaning OHLCV data (dictionary output)...")
    cleaner = CleanOHLCV(freq="D")  # Daily frequency
    cleaned_dict = cleaner.clean(ohlcv_dict, validate=True)
    
    # Print summary
    cleaner.print_summary(cleaned_dict)
    
    # Clean and merge
    print("\n🔗 Merging cleaned OHLCV data...")
    merged_df = cleaner.clean_and_merge(ohlcv_dict, validate=True)
    print(f"✓ Merged DataFrame shape: {merged_df.shape}")
    
    # Get unique symbols from MultiIndex columns
    symbols = merged_df.columns.get_level_values(0).unique().tolist()
    print(f"✓ Symbols ({len(symbols)}): {sorted(symbols)}")
    
    # Save cleaned data
    print("\n💾 Saving cleaned data...")
    clean_dir = project_root / "data" / "clean_ohlcv"
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol, df in cleaned_dict.items():
        filepath = clean_dir / f"{symbol}_clean.csv"
        df.to_csv(filepath)
        print(f"  ✓ Saved {symbol} → {filepath.name}")
    
    # Save merged data
    merged_path = clean_dir / "merged_ohlcv.csv"
    merged_df.to_csv(merged_path)
    print(f"  ✓ Saved merged data → merged_ohlcv.csv ({merged_df.shape[0]} rows × {merged_df.shape[1]} cols)")
    
    # Show sample
    print("\n" + "=" * 80)
    print("Sample Cleaned Data (first symbol)")
    print("=" * 80)
    first_symbol = sorted(cleaned_dict.keys())[0]
    print(f"\n{first_symbol}:")
    print(cleaned_dict[first_symbol].head(10))
    
    # Show sample merged data intelligently
    print("\n" + "=" * 80)
    print("Sample Merged Data (first 5 rows)")
    print("=" * 80)
    
    # Get data types to find OHLCV symbols
    data_types = cleaner.get_data_types()
    ohlcv_symbols = [s for s, dt in data_types.items() if dt == "ohlcv"]
    ts_symbols = [s for s, dt in data_types.items() if dt == "timeseries"]
    
    # Show OHLCV samples
    if ohlcv_symbols:
        first_ohlcv = sorted(ohlcv_symbols)[:2]
        ohlcv_cols = [(s, c) for s in first_ohlcv for c in ["Open", "Close"]]
        print(f"\nOHLCV Symbols ({first_ohlcv}):")
        print(merged_df[ohlcv_cols].head(5))
    
    # Show time-series samples
    if ts_symbols:
        first_ts = sorted(ts_symbols)[:2]
        ts_cols = [(s, "Value") for s in first_ts]
        print(f"\nTime-Series Symbols ({first_ts}):")
        print(merged_df[ts_cols].head(5))
    
    # Data type summary
    print("\n" + "=" * 80)
    print("Data Types Detected")
    print("=" * 80)
    print(f"✓ OHLCV data: {len(ohlcv_symbols)} symbols")
    if ohlcv_symbols:
        print(f"    {', '.join(sorted(ohlcv_symbols))}")
    print(f"✓ Time-series data: {len(ts_symbols)} symbols")
    if ts_symbols:
        print(f"    {', '.join(sorted(ts_symbols))}")
    
    # Alignment info
    print("\n" + "=" * 80)
    print("Alignment Info")
    print("=" * 80)
    min_date = merged_df.index.min()
    max_date = merged_df.index.max()
    print(f"✓ Date range: {min_date.date()} to {max_date.date()}")
    print(f"✓ Total days: {len(merged_df)}")
    print(f"✓ Frequency: Daily (D)")


if __name__ == "__main__":
    main()