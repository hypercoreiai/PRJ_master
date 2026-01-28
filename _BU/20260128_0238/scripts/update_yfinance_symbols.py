"""
Update yfinance symbols by:
1. Converting Kraken symbols (e.g. BTC/USD) to yfinance format (BTC-USD)
2. Testing availability on yfinance
3. Adding available symbols to yfinance symbols file
4. Prepending indicators (if they exist on yfinance) to the top
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yfinance as yf


def read_symbols(filepath: Path) -> set[str]:
    """Read symbols from a file, one per line, stripping whitespace."""
    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        return set()
    
    with open(filepath, "r") as f:
        symbols = {line.strip() for line in f if line.strip()}
    
    print(f"✓ Read {len(symbols)} symbols from {filepath.name}")
    return symbols


def write_symbols(filepath: Path, symbols: list[str], mode: str = "w") -> None:
    """Write symbols to a file, one per line."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode) as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    print(f"✓ Wrote {len(symbols)} symbols to {filepath.name}")


def kraken_to_yfinance(kraken_symbol: str) -> str:
    """Convert Kraken symbol (e.g. BTC/USD) to yfinance format (BTC-USD)."""
    return kraken_symbol.replace("/", "-")


def is_symbol_available_yfinance(symbol: str) -> bool:
    """Test if a symbol is available on yfinance."""
    try:
        # Try to fetch 1 day of data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        return not hist.empty
    except Exception as e:
        return False


def main():
    """Main script logic."""
    print("=" * 70)
    print("Updating yfinance symbols from Kraken and Indicators")
    print("=" * 70)
    
    # Define paths
    kraken_path = project_root / "src" / "symbols" / "kraken"
    indicators_path = project_root / "src" / "symbols" / "indicators"
    yfinance_path = project_root / "src" / "symbols" / "yfinance"
    
    # Read existing yfinance symbols
    existing_yf_symbols = read_symbols(yfinance_path)
    print(f"📋 Existing yfinance symbols: {len(existing_yf_symbols)}")
    
    # Process indicators (add to top)
    print("\n" + "=" * 70)
    print("Processing indicators...")
    print("=" * 70)
    indicator_symbols = read_symbols(indicators_path)
    available_indicators = []
    
    for indicator in sorted(indicator_symbols):
        if is_symbol_available_yfinance(indicator):
            available_indicators.append(indicator)
            print(f"✓ {indicator} - available on yfinance")
        else:
            print(f"✗ {indicator} - NOT available on yfinance")
    
    # Process Kraken symbols
    print("\n" + "=" * 70)
    print("Processing Kraken symbols...")
    print("=" * 70)
    kraken_symbols = read_symbols(kraken_path)
    available_kraken = []
    
    for kraken_sym in sorted(kraken_symbols):
        yf_symbol = kraken_to_yfinance(kraken_sym)
        
        # Skip if already in yfinance
        if yf_symbol in existing_yf_symbols:
            print(f"⊘ {yf_symbol} - already in yfinance")
            continue
        
        # Test availability
        if is_symbol_available_yfinance(yf_symbol):
            available_kraken.append(yf_symbol)
            print(f"✓ {kraken_sym} → {yf_symbol} - available on yfinance")
        else:
            print(f"✗ {kraken_sym} → {yf_symbol} - NOT available on yfinance")
    
    # Combine: indicators first, then kraken, then existing
    print("\n" + "=" * 70)
    print("Combining symbols...")
    print("=" * 70)
    
    combined_symbols = available_indicators + available_kraken + sorted(existing_yf_symbols)
    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for sym in combined_symbols:
        if sym not in seen:
            unique_symbols.append(sym)
            seen.add(sym)
    
    print(f"📊 Indicators to add: {len(available_indicators)}")
    print(f"📊 Kraken symbols to add: {len(available_kraken)}")
    print(f"📊 Existing symbols: {len(existing_yf_symbols)}")
    print(f"📊 Total unique symbols: {len(unique_symbols)}")
    
    # Write updated symbols
    if unique_symbols:
        write_symbols(yfinance_path, unique_symbols, mode="w")
        print(f"\n✓ Successfully updated {yfinance_path}")
    else:
        print(f"\n⚠️  No symbols to write!")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"New indicators added: {len(available_indicators)}")
    print(f"New Kraken symbols added: {len(available_kraken)}")
    print(f"Total symbols in yfinance file: {len(unique_symbols)}")
    
    if available_indicators:
        print(f"\nIndicators added:")
        for sym in available_indicators:
            print(f"  - {sym}")
    
    if available_kraken:
        print(f"\nKraken symbols added:")
        for sym in available_kraken:
            print(f"  - {sym}")


if __name__ == "__main__":
    main()