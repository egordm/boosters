#!/usr/bin/env python3
"""Generate real-world benchmark datasets as parquet files.

This script downloads and preprocesses standard ML benchmark datasets:
- California Housing (regression)
- Adult/Census Income (binary classification)
- Covertype (multiclass classification)

Usage:
    cd tools/data_generation
    uv run python scripts/generate_benchmark_datasets.py

Output:
    data/benchmarks/
        california_housing.parquet
        adult.parquet
        covertype.parquet

The parquet files use a standardized format:
- `y`: float32 target column
- `x_0`, `x_1`, ..., `x_N`: float32 feature columns
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"


def ensure_dependencies():
    """Check that required packages are available."""
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: uv add pandas pyarrow scikit-learn")
        sys.exit(1)


def standardize_dataframe(df, target_col: str):
    """Convert a dataframe to standardized format with y and x_0, x_1, ... columns."""
    import pandas as pd
    
    # Extract target
    y = df[target_col].astype('float32')
    
    # Get feature columns (everything except target)
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Build new dataframe with standardized column names
    result = pd.DataFrame({'y': y})
    for i, col in enumerate(feature_cols):
        result[f'x_{i}'] = df[col].astype('float32')
    
    return result


def generate_california_housing():
    """California Housing dataset (regression).
    
    20,640 samples, 8 features.
    Target: median house value (in $100,000s).
    """
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    
    print("Generating California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    
    # Combine features and target
    df = data.frame.copy()
    df = standardize_dataframe(df, 'MedHouseVal')
    
    output_path = DATA_DIR / "california_housing.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len([c for c in df.columns if c.startswith('x_')])}")
    print(f"  Target range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    return output_path


def generate_adult():
    """Adult/Census Income dataset (binary classification).
    
    48,842 samples, 14 features (6 numeric, 8 categorical -> one-hot encoded).
    Target: income >50K (1) or <=50K (0).
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml
    
    print("Generating Adult dataset...")
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    
    df = data.frame.copy()
    
    # Convert target to binary (0/1)
    df['target'] = (df['class'] == '>50K').astype('float32')
    df = df.drop(columns=['class'])
    
    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype='float32')
    
    # Convert all to float32
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].astype('float32')
    
    df = standardize_dataframe(df, 'target')
    
    output_path = DATA_DIR / "adult.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len([c for c in df.columns if c.startswith('x_')])}")
    print(f"  Class distribution: {df['y'].value_counts().to_dict()}")
    return output_path


def generate_covertype():
    """Covertype dataset (multiclass classification).
    
    581,012 samples, 54 features.
    Target: 7 forest cover types (classes 1-7, we remap to 0-6).
    """
    import pandas as pd
    from sklearn.datasets import fetch_covtype
    
    print("Generating Covertype dataset...")
    data = fetch_covtype(as_frame=True)
    
    df = data.frame.copy()
    # Target is 'Cover_Type', values 1-7. Remap to 0-6.
    df['target'] = (df['Cover_Type'] - 1).astype('float32')
    df = df.drop(columns=['Cover_Type'])
    
    # Convert all to float32
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].astype('float32')
    
    df = standardize_dataframe(df, 'target')
    
    output_path = DATA_DIR / "covertype.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len([c for c in df.columns if c.startswith('x_')])}")
    print(f"  Classes: {sorted(df['y'].unique())}")
    return output_path


def main():
    ensure_dependencies()
    
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {DATA_DIR}")
    print()
    
    paths = []
    paths.append(generate_california_housing())
    print()
    paths.append(generate_adult())
    print()
    paths.append(generate_covertype())
    print()
    
    print("=" * 60)
    print("Generated datasets:")
    for p in paths:
        print(f"  {p}")
    print()
    print("Use with quality_benchmark:")
    print("  cargo run --bin quality_benchmark --release \\")
    print("      --features 'bench-xgboost,bench-lightgbm,io-parquet'")


if __name__ == "__main__":
    main()
