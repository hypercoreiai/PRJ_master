"""
Clean and standardize OHLCV data and time-series indicators from multiple sources.

Aligns multiple DataFrames to a common date range, fills gaps using
forward-fill then back-fill, validates OHLCV constraints (when applicable),
and provides both dictionary and merged output formats.

Handles:
- OHLCV data (Open, High, Low, Close, Volume)
- Time-series indicators (single value column)
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np


class CleanOHLCV:
    """
    Clean and standardize OHLCV data and time-series indicators.
    
    Features:
    - Aligns all symbols to union of date ranges
    - Detects data type (OHLCV vs time-series)
    - Forward-fills (ffill) then back-fills (bfill) gaps
    - Fills volume with 0 for missing periods
    - Validates OHLCV constraints (High >= Low, etc.)
    - Outputs as dictionary or merged DataFrame
    """
    
    # Standard OHLCV columns
    OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    
    def __init__(self, freq: str = "B"):
        """
        Initialize CleanOHLCV.
        
        Parameters
        ----------
        freq : str, default "B"
            Frequency for output index ("D" = daily, "B" = business days)
        """
        self.freq = freq
        self.validation_errors = {}
        self.data_types = {}  # Track which symbols are OHLCV vs time-series
    
    def clean(
        self,
        data_dict: dict[str, pd.DataFrame],
        freq: Optional[str] = None,
        validate: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Clean OHLCV and time-series data and return as dictionary.
        
        Parameters
        ----------
        data_dict : dict[str, pd.DataFrame]
            Dictionary of DataFrames (OHLCV or time-series), keyed by symbol
        freq : str, optional
            Frequency override ("D" or "B"). Uses self.freq if None.
        validate : bool, default True
            Whether to validate OHLCV constraints (for OHLCV data only)
        
        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of cleaned DataFrames
        """
        freq = freq or self.freq
        
        # Get aligned date range
        aligned_index = self._get_aligned_index(data_dict, freq)
        
        # Clean each symbol
        cleaned = {}
        for symbol, df in data_dict.items():
            cleaned[symbol] = self._clean_single(df, symbol, aligned_index, validate)
        
        return cleaned
    
    def clean_and_merge(
        self,
        data_dict: dict[str, pd.DataFrame],
        freq: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Clean data and return as merged DataFrame.
        
        Parameters
        ----------
        data_dict : dict[str, pd.DataFrame]
            Dictionary of DataFrames (OHLCV or time-series), keyed by symbol
        freq : str, optional
            Frequency override ("D" or "B"). Uses self.freq if None.
        validate : bool, default True
            Whether to validate OHLCV constraints
        
        Returns
        -------
        pd.DataFrame
            Merged DataFrame with MultiIndex columns (Symbol, Column)
        """
        cleaned_dict = self.clean(data_dict, freq=freq, validate=validate)
        
        # Merge into single DataFrame with MultiIndex columns
        dfs = []
        for symbol, df in sorted(cleaned_dict.items()):
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
            dfs.append(df)
        
        merged = pd.concat(dfs, axis=1)
        return merged
    
    def _detect_data_type(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Detect if data is OHLCV or time-series.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze
        symbol : str
            Symbol name
        
        Returns
        -------
        str
            "ohlcv" or "timeseries"
        """
        # Check for OHLCV columns
        ohlcv_cols = [c for c in self.OHLCV_COLUMNS if c in df.columns]
        
        if len(ohlcv_cols) >= 4:  # At least Open, High, Low, Close
            data_type = "ohlcv"
        else:
            data_type = "timeseries"
        
        self.data_types[symbol] = data_type
        return data_type
    
    def _get_aligned_index(
        self,
        data_dict: dict[str, pd.DataFrame],
        freq: str
    ) -> pd.DatetimeIndex:
        """
        Get aligned DatetimeIndex (union of all date ranges).
        
        Parameters
        ----------
        data_dict : dict[str, pd.DataFrame]
            Dictionary of DataFrames
        freq : str
            Frequency for index
        
        Returns
        -------
        pd.DatetimeIndex
            Aligned DatetimeIndex covering all data
        """
        if not data_dict:
            raise ValueError("Empty data dictionary")
        
        # Find earliest start and latest end
        min_date = min(df.index.min() for df in data_dict.values())
        max_date = max(df.index.max() for df in data_dict.values())
        
        # Create continuous index
        aligned_index = pd.date_range(start=min_date, end=max_date, freq=freq)
        
        return aligned_index
    
    def _clean_single(
        self,
        df: pd.DataFrame,
        symbol: str,
        aligned_index: pd.DatetimeIndex,
        validate: bool
    ) -> pd.DataFrame:
        """
        Clean a single DataFrame (OHLCV or time-series).
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame
        symbol : str
            Symbol name (for error reporting)
        aligned_index : pd.DatetimeIndex
            Target aligned index
        validate : bool
            Whether to validate constraints
        
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: Index must be DatetimeIndex")
        
        # Detect data type
        data_type = self._detect_data_type(df, symbol)
        
        # Reindex to aligned index
        df = df.reindex(aligned_index)
        
        if data_type == "ohlcv":
            df = self._clean_ohlcv(df, symbol, validate)
        else:
            df = self._clean_timeseries(df, symbol)
        
        return df
    
    def _clean_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        validate: bool
    ) -> pd.DataFrame:
        """
        Clean OHLCV data.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
        symbol : str
            Symbol name
        validate : bool
            Whether to validate constraints
        
        Returns
        -------
        pd.DataFrame
            Cleaned OHLCV DataFrame
        """
        # Identify OHLC and Volume columns
        ohlc_cols = [c for c in self.OHLCV_COLUMNS[:-1] if c in df.columns]
        vol_col = "Volume" if "Volume" in df.columns else None
        
        # Fill OHLC: ffill then bfill
        for col in ohlc_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # Fill Volume with 0 (no trading = no volume)
        if vol_col and vol_col in df.columns:
            df[vol_col] = df[vol_col].fillna(0)
        
        # Validate OHLCV constraints
        if validate:
            self._validate_ohlcv(df, symbol, ohlc_cols)
        
        # Ensure column order (keep only available columns)
        available_cols = [c for c in self.OHLCV_COLUMNS if c in df.columns]
        df = df[available_cols]
        
        return df
    
    def _clean_timeseries(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Clean time-series indicator data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Time-series DataFrame
        symbol : str
            Symbol name
        
        Returns
        -------
        pd.DataFrame
            Cleaned time-series DataFrame
        """
        # Fill all numeric columns: ffill then bfill
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].ffill().bfill()
        
        return df
    
    def _validate_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        ohlc_cols: list[str]
    ) -> None:
        """
        Validate OHLCV constraints.
        
        Checks:
        - High >= Low
        - High >= Close
        - Low <= Close
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame to validate
        symbol : str
            Symbol name (for error reporting)
        ohlc_cols : list[str]
            List of OHLC column names
        """
        if not ohlc_cols or len(ohlc_cols) < 4:
            return  # Skip if incomplete OHLC data
        
        errors = []
        
        # High >= Low
        if "High" in df.columns and "Low" in df.columns:
            bad_hl = df[df["High"] < df["Low"]]
            if not bad_hl.empty:
                errors.append(f"High < Low: {len(bad_hl)} rows")
        
        # High >= Close
        if "High" in df.columns and "Close" in df.columns:
            bad_hc = df[df["High"] < df["Close"]]
            if not bad_hc.empty:
                errors.append(f"High < Close: {len(bad_hc)} rows")
        
        # Low <= Close
        if "Low" in df.columns and "Close" in df.columns:
            bad_lc = df[df["Low"] > df["Close"]]
            if not bad_lc.empty:
                errors.append(f"Low > Close: {len(bad_lc)} rows")
        
        if errors:
            self.validation_errors[symbol] = errors
    
    def get_validation_errors(self) -> dict[str, list[str]]:
        """
        Get validation errors from last clean operation.
        
        Returns
        -------
        dict[str, list[str]]
            Dictionary of validation errors by symbol
        """
        return self.validation_errors
    
    def get_data_types(self) -> dict[str, str]:
        """
        Get detected data types from last clean operation.
        
        Returns
        -------
        dict[str, str]
            Dictionary of data types (ohlcv or timeseries) by symbol
        """
        return self.data_types
    
    def print_summary(self, cleaned_dict: dict[str, pd.DataFrame]) -> None:
        """
        Print summary of cleaned data.
        
        Parameters
        ----------
        cleaned_dict : dict[str, pd.DataFrame]
            Dictionary of cleaned DataFrames
        """
        print("=" * 100)
        print("Data Cleaning Summary")
        print("=" * 100)
        
        if not cleaned_dict:
            print("⚠️  No data to clean")
            return
        
        print(f"\n✓ Successfully cleaned {len(cleaned_dict)} symbols\n")
        print("Symbol Statistics:")
        print("-" * 100)
        print(f"{'Symbol':<15} {'Type':<12} {'Rows':<8} {'Start Date':<15} {'End Date':<15} {'Columns':<40}")
        print("-" * 100)
        
        for symbol, df in sorted(cleaned_dict.items()):
            start = df.index.min().date()
            end = df.index.max().date()
            cols = ", ".join(df.columns.tolist())
            data_type = self.data_types.get(symbol, "unknown")
            print(f"{symbol:<15} {data_type:<12} {len(df):<8} {str(start):<15} {str(end):<15} {cols:<40}")
        
        # Print validation errors if any
        if self.validation_errors:
            print("\n⚠️  Validation Warnings:")
            print("-" * 100)
            for symbol, errors in self.validation_errors.items():
                print(f"  {symbol}:")
                for error in errors:
                    print(f"    - {error}")
        else:
            print("\n✓ All OHLCV constraints validated")
        
        print()