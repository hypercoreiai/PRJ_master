# OHLCV & Ticker — Reference

Four classes in `.\src\ohlcv\` for downloading and cleaning price and ticker data from **yfinance** (OHLCV) and **Kraken** (OHLCV and Ticker). All use **symbol** and **period**; the date range is always **[current − period, current]**.

---

## Shared behavior

- **Period** — Strings like `"3y"`, `"2y"`, `"7y"` (number + `"y"`). Parsed as years from "now".
- **Date range** — `start = current − period`, `end = current` (inclusive of "today").

---

## 1. YFinanceOHLCV — `yfinance_ohlcv.py`

Downloads **OHLCV** (Open, High, Low, Close, Volume) from yfinance.

| Argument   | Description |
|-----------|--------------|
| `symbol`  | Ticker (e.g. `"AAPL"`, `"BTC-USD"`, `"^GSPC"`). |
| `period`  | Lookback: `"3y"`, `"2y"`, `"7y"`, etc. |

**Returns:** `pd.DataFrame` with columns Open, High, Low, Close, Volume (and Dividends, Stock Splits when present). Index: DatetimeIndex named `"date"`.

**Example:** `YFinanceOHLCV().get("AAPL", "3y")`

---

## 2. KrakenOHLCV — `kraken_ohlcv.py`

Downloads **OHLCV** from Kraken REST `GET /0/public/OHLC`. Paginates when the range needs more than 720 bars.

| Argument        | Description |
|-----------------|-------------|
| `symbol`        | Kraken pair (e.g. `"XBTUSD"`, `"XXBTZUSD"`, `"ETHUSD"`). |
| `period`        | Lookback: `"3y"`, `"2y"`, `"7y"`, etc. |
| `interval_min`  | Bar length in minutes. Allowed: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600. Default **60** (1 h). |

**Returns:** `pd.DataFrame` with columns open, high, low, close, vwap, volume, count. Index: DatetimeIndex named `"date"`.

**Example:** `KrakenOHLCV().get("XBTUSD", "2y", interval_min=1440)` for 1‑day bars.

---

## 3. KrakenTicker — `kraken_ticker.py`

Downloads **ticker** (current bid/ask/last/volume, etc.) from Kraken REST `GET /0/public/Ticker`. Snapshot as of the request time; `period` is for a consistent API (end of range = "now").

| Argument | Description |
|----------|-------------|
| `symbol` | Kraken pair (e.g. `"XBTUSD"`, `"XXBTZUSD"`). |
| `period` | Lookback string for API consistency; snapshot is at current time. |

**Returns:** `pd.DataFrame` with one row. Columns include `date`, `pair`, and flattened ticker fields (e.g. `a` / `a_lot` = ask, `b` / `b_lot` = bid, `c` / `c_lot` = last).

**Example:** `KrakenTicker().get("XBTUSD", "1y")`

---

## 4. CleanOHLCV — `clean_ohlcv.py`

Cleans and standardizes **OHLCV and time-series indicator data** from multiple sources. Aligns all symbols to a common date range, fills gaps, validates OHLCV constraints, and provides both dictionary and merged output formats.

| Argument    | Description |
|-------------|-------------|
| `freq`      | Output frequency: `"D"` (daily) or `"B"` (business days). Default **`"B"`**. |

### Methods

#### `clean(data_dict, freq=None, validate=True)`

Clean data and return as dictionary of DataFrames.

| Argument    | Description |
|-------------|-------------|
| `data_dict` | `dict[str, pd.DataFrame]` — Dictionary of OHLCV or time-series DataFrames, keyed by symbol. |
| `freq`      | Frequency override. Uses instance `freq` if `None`. |
| `validate`  | Whether to validate OHLCV constraints (High ≥ Low, etc.). Default **`True`**. |

**Returns:** `dict[str, pd.DataFrame]` — Cleaned DataFrames, aligned to union of all date ranges.

**Example:**
```python
cleaner = CleanOHLCV(freq="D")
cleaned = cleaner.clean({
    "AAPL": df1,
    "BTC-USD": df2,
    "INDPRO": df3
})
```

#### `clean_and_merge(data_dict, freq=None, validate=True)`

Clean data and return as merged DataFrame with MultiIndex columns (Symbol, Column).

| Argument    | Description |
|-------------|-------------|
| `data_dict` | `dict[str, pd.DataFrame]` — Dictionary of DataFrames. |
| `freq`      | Frequency override. Uses instance `freq` if `None`. |
| `validate`  | Whether to validate OHLCV constraints. Default **`True`**. |

**Returns:** `pd.DataFrame` — Merged DataFrame with MultiIndex columns `(Symbol, Column)`.

**Example:**
```python
cleaner = CleanOHLCV(freq="D")
merged = cleaner.clean_and_merge({
    "AAPL": df1,
    "BTC-USD": df2
})
# Access data: merged[("AAPL", "Close")], merged[("BTC-USD", "Volume")]
```

### Features

- **Mixed data types** — Handles both OHLCV (stock prices) and time-series indicators (FRED, BEA).
- **Auto-detection** — Identifies data type based on columns (≥4 OHLCV columns = OHLCV, else time-series).
- **Gap filling** — Forward-fill (`ffill`) then back-fill (`bfill`) for prices; fills Volume with 0.
- **Index alignment** — All symbols aligned to union of date ranges (earliest start to latest end).
- **Validation** — Checks High ≥ Low, High ≥ Close, Low ≤ Close (reports errors without stopping).
- **Diagnostic methods** — `get_validation_errors()`, `get_data_types()`, `print_summary()`.

### Gap Filling Strategy

| Data Type | Column        | Strategy |
|-----------|---------------|----------|
| OHLCV     | Open, High, Low, Close | `ffill().bfill()` |
| OHLCV     | Volume        | `fillna(0)` |
| Time-series | Numeric columns | `ffill().bfill()` |

### Validation Constraints (OHLCV only)

- **High ≥ Low** — High price must not be below low price.
- **High ≥ Close** — High price must not be below close price.
- **Low ≤ Close** — Low price must not be above close price.

Violations are logged but do not prevent cleaning. Access via `get_validation_errors()`.

---

## Usage

```python
import sys
sys.path.insert(0, "src")  # or path to project root / src

from ohlcv import YFinanceOHLCV, KrakenOHLCV, KrakenTicker, CleanOHLCV
import pandas as pd

# Download OHLCV data
y = YFinanceOHLCV()
df_aapl = y.get("AAPL", "3y")

k = KrakenOHLCV()
df_btc = k.get("XBTUSD", "2y", interval_min=1440)

# Clean and align multiple symbols
cleaner = CleanOHLCV(freq="D")

# As dictionary
cleaned_dict = cleaner.clean({
    "AAPL": df_aapl,
    "BTC-USD": df_btc
}, validate=True)

# As merged DataFrame
merged = cleaner.clean_and_merge({
    "AAPL": df_aapl,
    "BTC-USD": df_btc
})

# Access merged data
close_prices = merged[("AAPL", "Close")], merged[("BTC-USD", "Close")]

# View diagnostics
cleaner.print_summary(cleaned_dict)
print(cleaner.get_data_types())
print(cleaner.get_validation_errors())
```

**Note:** On some systems the package directory may appear as `OHLCV` (uppercase); use `from OHLCV import ...` if `from ohlcv import ...` fails.

---

## File layout

- **`_period.py`** — `parse_period()`, `date_range_for_period()` (shared by all).
- **`yfinance_ohlcv.py`** — `YFinanceOHLCV`.
- **`kraken_ohlcv.py`** — `KrakenOHLCV` (uses `requests`; no API key for public OHLC).
- **`kraken_ticker.py`** — `KrakenTicker` (uses `requests`; no API key for public Ticker).
- **`clean_ohlcv.py`** — `CleanOHLCV` (data cleaning and alignment).

Requirements: `yfinance`, `pandas`, `requests`.

---

## Examples

### Download and Clean Crypto Data

```python
from ohlcv import KrakenOHLCV, CleanOHLCV

kraken = KrakenOHLCV()
data = {
    "BTC": kraken.get("XBTUSD", "1y", interval_min=1440),
    "ETH": kraken.get("ETHUSD", "1y", interval_min=1440)
}

cleaner = CleanOHLCV(freq="D")
cleaned = cleaner.clean(data)

for symbol, df in cleaned.items():
    print(f"{symbol}: {len(df)} days aligned")
```

### Download Stock and Indicator Data, Then Clean

```python
from ohlcv import YFinanceOHLCV, CleanOHLCV
from indicators import IndicatorFetcher
import pandas as pd

# Get stock and indicator data
yf = YFinanceOHLCV()
fetcher = IndicatorFetcher()

data = {
    "AAPL": yf.get("AAPL", "3y"),
    "SPY": yf.get("SPY", "3y"),
}

# Add indicators (as single-column time-series)
indicators_result = fetcher.get_indicators(["UNRATE", "UMCSENT"], period="3y")
for symbol, df in indicators_result.items():
    data[symbol] = df

# Clean and merge everything
cleaner = CleanOHLCV(freq="B")  # Business days
merged = cleaner.clean_and_merge(data)

print(f"Aligned {len(merged)} business days across {len(data)} assets")
merged.to_csv("aligned_data.csv")
```

### Validate Data Quality

```python
cleaner = CleanOHLCV()
cleaned = cleaner.clean(data_dict, validate=True)

errors = cleaner.get_validation_errors()
if errors:
    for symbol, error_list in errors.items():
        print(f"{symbol}: {error_list}")
```
