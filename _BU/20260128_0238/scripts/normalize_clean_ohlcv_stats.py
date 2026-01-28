"""
Normalize cleaned OHLCV data with detailed statistics.

Shows before/after statistics for each normalized column.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.normalize.revin import RevIN


def get_stats(data: np.ndarray) -> dict:
    """Compute statistics for data array, handling NaN values."""
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data.flatten()
    
    # Check if all values are NaN or zero
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "valid_count": 0,
        }
    
    return {
        "min": float(np.nanmin(valid_data)),
        "max": float(np.nanmax(valid_data)),
        "mean": float(np.nanmean(valid_data)),
        "std": float(np.nanstd(valid_data)),
        "median": float(np.nanmedian(valid_data)),
        "valid_count": len(valid_data),
    }


def normalize_ohlcv_file(
    filepath: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load, normalize, and return OHLCV data with statistics.
    
    Parameters
    ----------
    filepath : Path
        Path to cleaned OHLCV CSV file
    
    Returns
    -------
    tuple[pd.DataFrame, dict]
        (normalized_df, stats_dict)
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {filepath.name}: {len(df)} rows × {len(df.columns)} cols")
        
        # Get numeric columns to normalize
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            logger.warning(f"  No numeric columns found in {filepath.name}")
            return df, {}
        
        stats = {}
        
        # Prepare data: shape (num_samples, num_features)
        original_data = df[numeric_cols].values.astype(np.float32)
        
        # Get statistics before normalization
        for i, col in enumerate(numeric_cols):
            stats[col] = {
                "original": get_stats(original_data[:, i]),
            }
        
        # Convert to torch tensor: (1, num_samples, num_features)
        data_tensor = torch.from_numpy(original_data).unsqueeze(0)
        
        # Create RevIN with number of features
        num_features = len(numeric_cols)
        revin = RevIN(num_features=num_features)
        
        # Normalize
        normalized_tensor = revin(data_tensor)
        
        # Convert back to numpy
        normalized_data = normalized_tensor.squeeze(0).detach().numpy()
        
        # Add normalized columns and get after statistics
        for i, col in enumerate(numeric_cols):
            normalized_col_name = f"{col}_normalized"
            df[normalized_col_name] = normalized_data[:, i]
            stats[col]["normalized"] = get_stats(normalized_data[:, i])
            logger.debug(f"  Normalized column: {col}")
        
        return df, stats
    
    except Exception as e:
        logger.error(f"Failed to process {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return None, {}


def print_column_stats(col_name: str, stats: dict):
    """Print statistics for a normalized column."""
    orig = stats["original"]
    norm = stats["normalized"]
    
    # Check if data is valid
    if orig["valid_count"] == 0:
        print(f"\n  ⚠️  {col_name}:")
        print(f"     ⚠️  All values are NaN/invalid in original data - skipping statistics")
        return
    
    print(f"\n  📊 {col_name}:")
    print(f"     Before normalization (n={orig['valid_count']}):")
    print(f"       Min:    {orig['min']:15.6f}  →  After: {norm['min']:15.6f}")
    print(f"       Max:    {orig['max']:15.6f}  →  After: {norm['max']:15.6f}")
    print(f"       Mean:   {orig['mean']:15.6f}  →  After: {norm['mean']:15.6f}")
    print(f"       Std:    {orig['std']:15.6f}  →  After: {norm['std']:15.6f}")
    print(f"       Median: {orig['median']:15.6f}  →  After: {norm['median']:15.6f}")


def main():
    """Main normalization pipeline with statistics."""
    print("=" * 80)
    print("Normalizing Cleaned OHLCV Data (with Statistics)")
    print("=" * 80)
    
    # Directories
    clean_dir = project_root / "data" / "clean_ohlcv"
    output_dir = project_root / "data" / "normalized"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}\n")
    
    # Check for cleaned data
    csv_files = list(clean_dir.glob("*_clean.csv"))
    if not csv_files:
        logger.error(f"No cleaned OHLCV files found in {clean_dir}")
        return
    
    print(f"📂 Found {len(csv_files)} cleaned OHLCV files\n")
    
    # Process each file
    print("📥 Processing files...")
    print("=" * 80)
    
    successful = 0
    failed = 0
    data_quality_warnings = 0
    
    for filepath in sorted(csv_files):
        # Extract symbol from filename
        symbol = filepath.stem.rsplit("_", 1)[0]
        
        print(f"\n🔄 Processing {symbol}...")
        
        # Normalize data
        normalized_df, col_stats = normalize_ohlcv_file(filepath)
        
        if normalized_df is None:
            failed += 1
            print(f"  ✗ Failed to normalize")
            continue
        
        # Check for data quality issues
        has_nan_columns = False
        for col_name, stats_dict in col_stats.items():
            if stats_dict["original"]["valid_count"] == 0:
                has_nan_columns = True
                data_quality_warnings += 1
                break
        
        # Show column statistics
        for col_name, stats_dict in col_stats.items():
            print_column_stats(col_name, stats_dict)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{symbol}_{timestamp}_normalized.csv"
        output_path = output_dir / output_filename
        
        # Save normalized data
        try:
            normalized_df.to_csv(output_path)
            successful += 1
            
            normalized_cols = [c for c in normalized_df.columns if "_normalized" in c]
            
            status = "⚠️ " if has_nan_columns else "✓ "
            print(f"\n  {status}Saved to {output_filename}")
            print(f"    Shape: {normalized_df.shape[0]} rows × {normalized_df.shape[1]} cols")
            print(f"    Added: {len(normalized_cols)} normalized columns")
        
        except Exception as e:
            failed += 1
            logger.error(f"  ✗ {symbol}: Failed to save: {e}")
            print(f"  ✗ Failed to save")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 Normalization Complete")
    print("=" * 80)
    print(f"   ✓ Successful:           {successful}/{len(csv_files)}")
    print(f"   ⚠️  Data quality issues:  {data_quality_warnings}/{len(csv_files)}")
    print(f"   ✗ Failed:              {failed}/{len(csv_files)}")
    print(f"   📁 Output:             {output_dir}\n")


if __name__ == "__main__":
    main()