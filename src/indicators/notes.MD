# Indicators — Symbol Map & Reference

Reference for `IndicatorFetcher` in `get_indicators.py`: accepted indicator names, symbol maps, and usage details.

---

## Data Sources

| Source    | API key required | Stored in      |
|----------|-------------------|----------------|
| **yfinance** | No               | —              |
| **FRED**     | Yes              | `.env` as `FRED_API_KEY` |
| **BEA**      | Yes              | `.env` as `BEA_API_KEY` (or `BEA_KEY`) |

Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html  
Get a free BEA API key: https://apps.bea.gov/API/signup/

---

## FRED Symbol Map

Economic indicators from Federal Reserve Economic Data. Use either the **series ID** or an **alias** in `indicator_names`.

### By category

| Series ID       | Description                                             | Aliases (also accepted)   |
|-----------------|---------------------------------------------------------|---------------------------|
| **Interest rates** | | |
| `T10Y2Y`        | 10-Year/2-Year Treasury Spread                          | `treasury_spread_10y2y`   |
| `TB3MS`         | 3-Month Treasury Bill Rate                              | `treasury_3m`             |
| `MORTGAGE30US`  | 30-Year Conventional Mortgage Rate                      | `mortgage_30y`            |
| **Inflation & prices** | | |
| `CPIAUCSL`      | Consumer Price Index (All Urban)                        | `cpi`                     |
| `PCE`           | Personal Consumption Expenditures                       | `pce`                     |
| **Economic activity** | | |
| `GDPC1`         | Real Gross Domestic Product (level)                     | `gdp`                     |
| `A191RL1Q225SBEA` | Real GDP growth (percent change, annual)              | `gdp_growth`              |
| `INDPRO`        | Industrial Production Index                             | `industrial_production`   |
| `RSXFS`         | Advance Real Retail and Food Services Sales             | `retail_sales`            |
| **Labor market** | | |
| `UNRATE`        | Civilian Unemployment Rate                              | `unemployment`            |
| `PAYEMS`        | Total Nonfarm Payrolls                                   | `nonfarm_payrolls`        |
| **Financial & sentiment** | | |
| `BAMLC0A4CBBB`  | ICE BofA BBB US Corporate Index (proxy for corp spreads) | `corporate_bond_spread`   |
| `TDSP`          | Household Debt Service Payments (% of disposable income) | `household_debt_service`  |
| `UMCSENT`       | U. of Michigan Consumer Sentiment                        | `consumer_sentiment`      |

### Full FRED alias → series ID

```
treasury_spread_10y2y → T10Y2Y
treasury_3m           → TB3MS
mortgage_30y          → MORTGAGE30US
cpi                   → CPIAUCSL
pce                   → PCE
gdp_growth            → A191RL1Q225SBEA
gdp                   → GDPC1
industrial_production → INDPRO
retail_sales          → RSXFS
unemployment          → UNRATE
nonfarm_payrolls      → PAYEMS
corporate_bond_spread → BAMLC0A4CBBB
household_debt_service→ TDSP
consumer_sentiment    → UMCSENT
```

---

## BEA Symbol Map

Economic data from the Bureau of Economic Analysis (BEA) via **beaapi**. Uses NIPA (National Income and Product Accounts) tables. Use either the **indicator ID** or an **alias** in `indicator_names`. Requires `BEA_API_KEY` (or `BEA_KEY`) in `.env`.

### BEA indicators

| Indicator ID         | Description                                   | Aliases (also accepted)   |
|----------------------|-----------------------------------------------|---------------------------|
| `BEA_GDP_GROWTH`     | Real GDP percent change (Table 1.1.1, line 1) | `bea_gdp_growth`          |
| `BEA_PCE_PRICE`      | PCE price index (Table 2.3.4, line 1)         | `bea_pce_inflation`, `bea_pce_price` |
| `BEA_PERSONAL_INCOME`| Personal income (Table 2.1, line 1)           | `bea_personal_income`     |
| `BEA_DISPOSABLE_INCOME` | Disposable personal income (Table 2.1, line 2) | `bea_disposable_income` |

### BEA alias → indicator ID

```
bea_gdp_growth        → BEA_GDP_GROWTH
bea_pce_inflation     → BEA_PCE_PRICE
bea_pce_price         → BEA_PCE_PRICE
bea_personal_income   → BEA_PERSONAL_INCOME
bea_disposable_income → BEA_DISPOSABLE_INCOME
```

### BEA data details

- **Dataset:** NIPA. **Frequency:** Quarterly (`Q`). **Year range:** Derived from `start_date`–`end_date`.
- **PCE inflation / PCE price:** Table 2.3.4 (Price Indexes for PCE by Major Type of Product), line 1 = PCE price index.
- **Real GDP growth:** Table 1.1.1 (Percent Change From Preceding Period), line 1 = Gross domestic product.
- **Personal / Disposable income:** Table 2.1 (Personal Income and Its Disposition), line 1 = Personal income, line 2 = Disposable personal income.

---

## yfinance Index Symbol Map

Market indexes use fixed tickers; **no `ticker` argument** is needed. Pass these names in `indicator_names`.

| Requested name | yfinance ticker | Description      |
|----------------|------------------|------------------|
| `sp500`        | `^GSPC`          | S&P 500 Index    |
| `s&p 500`      | `^GSPC`          | S&P 500 Index    |
| `s&p500`       | `^GSPC`          | S&P 500 Index    |
| `vix`          | `^VIX`           | CBOE Volatility Index |

---

## yfinance Stock-Level Indicator Map

Stock valuation, fundamentals, technicals, and OHLCV **require** a `ticker` (e.g. `"AAPL"`). Use any of the names below in `indicator_names`; they are normalized to the canonical key.

### Valuation & fundamentals (single-point; one-row DataFrame)

| Canonical key     | Description           | Example aliases                          |
|-------------------|-----------------------|------------------------------------------|
| `pe`              | Price-to-Earnings     | `p/e`, `p_e`, `price_to_earnings`        |
| `forward_pe`       | Forward P/E           | `forward p/e`, `forward_p_e`             |
| `pb`              | Price-to-Book         | `p/b`, `price_to_book`                   |
| `ev_ebitda`       | EV/EBITDA             | `ev/ebitda`                              |
| `dividend_yield`  | Dividend Yield        | `dividend yield`                         |
| `peg_ratio`       | PEG Ratio             | `peg`, `peg ratio`                       |
| `eps`             | Earnings Per Share    | `earnings per share`                     |
| `revenue_growth`  | Revenue Growth        | `revenue growth`                        |
| `free_cash_flow`  | Free Cash Flow        | `free cash flow`, `fcf`                  |
| `debt_to_equity`  | Debt-to-Equity        | `debt to equity`                         |
| `roe`             | Return on Equity      | `return on equity`, `return_on_equity`   |
| `net_profit_margin`| Net Profit Margin     | `profit margin`                         |

### Technical & market (time series DataFrames)

| Canonical key | Description                    | Example aliases                          |
|---------------|--------------------------------|------------------------------------------|
| `sma`         | Simple Moving Average (20)     | `moving average`, `simple moving average`|
| `ema`         | Exponential Moving Average (20)| `exponential moving average`             |
| `rsi`         | Relative Strength Index (14)   | `relative strength index`                |
| `macd`        | MACD (12/26/9)                 | —                                        |
| `volume`      | Volume                         | —                                        |
| `bollinger`   | Bollinger Bands (lower, mid, upper) | `bollinger bands`, `bollinger_bands` |
| `historical`  | OHLC (Open, High, Low, Close)  | `prices`, `open high low close`          |
| `ohlcv`       | OHLC + Volume                  | `open high low close volume`             |

---

## Usage Summary

- **`get_indicators(indicator_names, *, start_date=None, end_date=None, ticker=None)`**  
  Returns `dict[str, pd.DataFrame]`; keys match the names you passed (after normalization for aliases).

- **FRED, BEA, and index names**  
  Use `indicator_names` only; no `ticker`. Example: `["UNRATE", "BEA_GDP_GROWTH", "sp500", "vix"]`.

- **Stock-level yfinance**  
  Pass the same names in `indicator_names` **and** set `ticker="AAPL"` (or another symbol). Example: `["pe", "rsi", "bollinger"], ticker="AAPL"`.

- **Defaults**  
  If `start_date` / `end_date` are omitted, the last 5 years up to today are used.

- **Output**  
  Each entry is a separate DataFrame: e.g. `result["UNRATE"]`, `result["sp500"]`, `result["rsi"]`. Structure depends on the indicator (single-row for P/E-style, time series for RSI/FRED/index, etc.).

---

## CleanIndicators (`clean_indicators.py`)

`CleanIndicators` standardizes indicator and index data over a fixed period using `IndicatorFetcher`:

- **Date range** — Always `[present - period, present]`. Pass `period` as `"3y"`, `"2y"`, `"7y"`, etc.
- **Continuous time index** — Builds a full daily (or `freq`) index over that range and reindexes each series to it, adding rows where there are gaps.
- **Fill** — Applies `.ffill()` then `.bfill()` on all data columns so new/missing rows get filled.
- **Point-in-time indicators** — Single-row metrics (e.g. P/E, EPS) are broadcast to the full index as constant series so every symbol has the same index.

### Usage

```python
from indicators import CleanIndicators

cleaner = CleanIndicators()
data = cleaner.get_clean_indicators(
    symbols=["UNRATE", "sp500", "vix"],
    period="3y",
)
# data["sp500"], data["vix"], data["UNRATE"] each have a continuous DatetimeIndex
# from (today - 3 years) to today, with gaps filled via ffill/bfill.
```

### Parameters

| Argument   | Description |
|-----------|-------------|
| `symbols` | List of indicator/symbol names (same as `IndicatorFetcher.get_indicators`). |
| `period`  | Lookback length: `"3y"`, `"2y"`, `"7y"` (number + `"y"`). |
| `ticker`  | Optional; required for yfinance stock-level indicators (e.g. `"pe"`, `"rsi"`). |
| `freq`    | Target frequency for the continuous index; default `"D"` (calendar day), use `"B"` for business days. |

---

## File layout

- **`get_indicators.py`** — `IndicatorFetcher`, `FRED_INDICATORS`, `FRED_ALIASES`, `BEA_INDICATORS`, `BEA_ALIASES`, `YF_INDEX_TICKERS`, `YF_STOCK_ALIASES`.
- **`clean_indicators.py`** — `CleanIndicators` (standardizes by period, continuous index, ffill/bfill).
- **`.env`** — Set `FRED_API_KEY=...` for FRED and `BEA_API_KEY=...` for BEA (see `.env.example` in project root).
