"""
Normalize cleaned OHLCV data using RevIN (Reversible Instance Normalization).

Loads cleaned OHLCV files from data/clean_ohlcv/, applies RevIN normalization,
and saves normalized data to data/normalized/.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.normalize.revin import RevIN


def normalize_ohlcv_file(filepath: Path) -> pd.DataFrame:
    """
    Load, normalize, and return OHLCV data.
    
    Parameters
    ----------
    filepath : Path
        Path to cleaned OHLCV CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized columns added
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {filepath.name}: {len(df)} rows × {len(df.columns)} cols")
        
        # Get numeric columns to normalize
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            logger.warning(f"  No numeric columns found in {filepath.name}")
            return df
        
        # Prepare data: shape (num_samples, num_features)
        original_data = df[numeric_cols].values.astype(np.float32)
        
        # Convert to torch tensor: (1, num_samples, num_features)
        data_tensor = torch.from_numpy(original_data).unsqueeze(0)
        
        # Create RevIN with number of features
        num_features = len(numeric_cols)
        revin = RevIN(num_features=num_features)
        
        # Normalize
        logger.debug(f"  Input shape: {data_tensor.shape}")
        normalized_tensor = revin(data_tensor)
        logger.debug(f"  Normalized shape: {normalized_tensor.shape}")
        
        # Convert back to numpy
        normalized_data = normalized_tensor.squeeze(0).detach().numpy()
        
        # Add normalized columns with "_normalized" suffix
        for i, col in enumerate(numeric_cols):
            normalized_col_name = f"{col}_normalized"
            df[normalized_col_name] = normalized_data[:, i]
            logger.debug(f"  Normalized column: {col}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to process {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main normalization pipeline."""
    print("=" * 80)
    print("Normalizing Cleaned OHLCV Data")
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
    print("-" * 80)
    
    successful = 0
    failed = 0
    
    for filepath in sorted(csv_files):
        # Extract symbol from filename (e.g., "BTC-USD_clean.csv" -> "BTC-USD")
        symbol = filepath.stem.rsplit("_", 1)[0]
        
        # Normalize data
        normalized_df = normalize_ohlcv_file(filepath)
        
        if normalized_df is None:
            failed += 1
            print(f"✗ {symbol:20s} - Failed to normalize")
            continue
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{symbol}_{timestamp}_normalized.csv"
        output_path = output_dir / output_filename
        
        # Save normalized data
        try:
            normalized_df.to_csv(output_path)
            successful += 1
            
            # Get original column count
            orig_cols = pd.read_csv(filepath, index_col=0, nrows=0).shape[1]
            normalized_cols = [c for c in normalized_df.columns if "_normalized" in c]
            
            print(f"✓ {symbol:20s} → {output_filename}")
            print(f"  Rows: {normalized_df.shape[0]:6d} | Cols: {normalized_df.shape[1]:3d} (original: {orig_cols} + {len(normalized_cols)} normalized)")
        
        except Exception as e:
            failed += 1
            logger.error(f"  ✗ {symbol}: Failed to save: {e}")
            print(f"✗ {symbol:20s} - Failed to save")
    
    # Summary
    print("-" * 80)
    print(f"\n📊 Normalization Complete")
    print(f"   ✓ Successful: {successful}/{len(csv_files)}")
    print(f"   ✗ Failed:    {failed}/{len(csv_files)}")
    print(f"   📁 Output:   {output_dir}\n")


if __name__ == "__main__":
    main()