# OHLCV & Ticker — Reference

Three classes in `.\src\ohlcv\` for downloading price and ticker data from **yfinance** (OHLCV) and **Kraken** (OHLCV and Ticker). All use **symbol** and **period**; the date range is always **[current − period, current]**.

---

## Shared behavior

- **Period** — Strings like `"3y"`, `"2y"`, `"7y"` (number + `"y"`). Parsed as years from “now”.
- **Date range** — `start = current − period`, `end = current` (inclusive of “today”).

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

Downloads **ticker** (current bid/ask/last/volume, etc.) from Kraken REST `GET /0/public/Ticker`. Snapshot as of the request time; `period` is for a consistent API (end of range = “now”).

| Argument | Description |
|----------|-------------|
| `symbol` | Kraken pair (e.g. `"XBTUSD"`, `"XXBTZUSD"`). |
| `period` | Lookback string for API consistency; snapshot is at current time. |

**Returns:** `pd.DataFrame` with one row. Columns include `date`, `pair`, and flattened ticker fields (e.g. `a` / `a_lot` = ask, `b` / `b_lot` = bid, `c` / `c_lot` = last).

**Example:** `KrakenTicker().get("XBTUSD", "1y")`

---

## Usage

```python
import sys
sys.path.insert(0, "src")  # or path to project root / src

from ohlcv import YFinanceOHLCV, KrakenOHLCV, KrakenTicker

# yfinance OHLCV
y = YFinanceOHLCV()
df = y.get("AAPL", "3y")

# Kraken OHLCV (e.g. daily bars)
k = KrakenOHLCV()
df2 = k.get("XBTUSD", "2y", interval_min=1440)

# Kraken ticker (current snapshot)
t = KrakenTicker()
df3 = t.get("XBTUSD", "1y")
```

**Note:** On some systems the package directory may appear as `OHLCV` (uppercase); use `from OHLCV import ...` if `from ohlcv import ...` fails.

---

## File layout

- **`_period.py`** — `parse_period()`, `date_range_for_period()` (shared by all three).
- **`yfinance_ohlcv.py`** — `YFinanceOHLCV`.
- **`kraken_ohlcv.py`** — `KrakenOHLCV` (uses `requests`; no API key for public OHLC).
- **`kraken_ticker.py`** — `KrakenTicker` (uses `requests`; no API key for public Ticker).

Requirements: `yfinance`, `pandas`, `requests`.
