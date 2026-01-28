# Financial Data Pipeline Project

A comprehensive Python project for fetching, cleaning, and normalizing financial time-series data including OHLCV (Open, High, Low, Close, Volume) data and economic indicators.

## Overview

This project provides a modular pipeline for collecting financial data from multiple sources, standardizing it to consistent formats, and applying reversible normalization techniques. It's designed for machine learning applications requiring clean, normalized time-series data.

## Key Features

- **Multi-Source Data Fetching**: Collect OHLCV data from Kraken (crypto) and Yahoo Finance (stocks/indices)
- **Economic Indicators**: Fetch macroeconomic data from FRED (Federal Reserve) and BEA (Bureau of Economic Analysis)
- **Data Cleaning**: Standardize time indices, handle missing data, and ensure data quality
- **Reversible Normalization**: Implement RevIN (Reversible Instance Normalization) for ML preprocessing
- **Modular Architecture**: Clean separation of concerns with dedicated modules for each data type

## Project Structure

```
├── data/
│   ├── clean_indicators/     # Cleaned economic indicators (CSV)
│   ├── clean_ohlcv/          # Cleaned OHLCV data (CSV)
│   ├── normalized/           # Normalized data (CSV)
│   └── raw/                  # Raw downloaded data (CSV)
├── scripts/                  # Executable scripts for data operations
├── src/                     # Source code modules
│   ├── indicators/          # Economic indicators fetching & cleaning
│   ├── ohlcv/              # OHLCV data fetching (Kraken, yfinance)
│   ├── normalize/           # RevIN normalization transforms
│   ├── symbols/             # Symbol lists for different sources
│   └── models/              # (Reserved for ML models)
├── tests/                   # Unit tests
├── _BU/                     # Backup folders with timestamps
└── _Notes_MD/               # Documentation
```

## Data Sources

### OHLCV Data
- **Kraken**: Cryptocurrency pairs (e.g., XBTUSD, ETHUSD)
- **Yahoo Finance**: Stocks, ETFs, indices (e.g., AAPL, ^GSPC)

### Economic Indicators
- **FRED (Federal Reserve)**: Unemployment, CPI, GDP growth, Treasury spreads, etc.
- **BEA (Bureau of Economic Analysis)**: GDP growth, national accounts data

## Dependencies

- pandas >= 2.0.0
- yfinance >= 0.2.0
- fredapi >= 0.5.0
- beaapi >= 0.1.0
- python-dotenv >= 1.0.0
- requests >= 2.28.0
- pytest >= 7.0.0

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys in `.env` file:
   - `FRED_API_KEY`: Get from https://fred.stlouisfed.org/docs/api/api_key.html
   - `BEA_API_KEY`: Get from https://apps.bea.gov/API/signup/

## Usage

### Data Collection Scripts

- `get_rnd_data.py`: Download random samples from all sources
- `get_clean_rnd_indicators.py`: Fetch and clean economic indicators
- `normalize_clean_ohlcv.py`: Apply normalization to OHLCV data

### Core Modules

#### Indicators
```python
from src.indicators import IndicatorFetcher
fetcher = IndicatorFetcher()
data = fetcher.get("UNRATE", "5y")  # Unemployment rate, 5 years
```

#### OHLCV
```python
from src.ohlcv import KrakenOHLCV, YFinanceOHLCV
kraken = KrakenOHLCV()
data = kraken.get("XBTUSD", "2y")  # Bitcoin USD, 2 years

yfinance = YFinanceOHLCV()
data = yfinance.get("AAPL", "3y")  # Apple stock, 3 years
```

#### Normalization
```python
from src.normalize import RevinTransform
transformer = RevinTransform(num_features=4)  # OHLC features
normalized_df = transformer.fit_transform(df)
```

## Testing

Run tests with: `pytest`

Test coverage includes:
- Data fetching functionality
- Cleaning and standardization
- Normalization transforms
- API integrations

## Data Format Standards

- **Time Index**: Daily frequency, timezone-naive DatetimeIndex
- **Missing Data**: Forward/backward filled where appropriate
- **File Naming**: `{symbol}_{date}.csv` format
- **Normalization**: RevIN applied per feature, with mean/stdev preserved for reversal

## Contributing

1. Follow the modular structure
2. Add tests for new functionality
3. Update documentation in `_Notes_MD/`
4. Ensure data quality standards are maintained

## License

[Add license information here]