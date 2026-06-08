"""
Shared data-loading helpers for the SQLi detection pipeline.

These utilities keep path handling and column validation in one place for
notebooks, training scripts, and future tests.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "trainingdata.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

QUERY_COLUMN = "Query"
LABEL_COLUMN = "Label"


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw labelled SQL query dataset."""
    df = pd.read_csv(path)
    return _validate_query_label_columns(df, source=path)


def load_split(name: str, processed_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """Load one processed CSV split: train, val, test, or cleaned."""
    valid_names = {"train", "val", "test", "cleaned"}
    if name not in valid_names:
        raise ValueError(f"name must be one of {sorted(valid_names)}")

    path = Path(processed_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    return _validate_query_label_columns(df, source=path)


def load_feature_split(
    name: str,
    processed_dir: Path | str = PROCESSED_DIR,
) -> Tuple[object, np.ndarray]:
    """Load one generated sparse feature matrix and its labels."""
    valid_names = {"train", "val", "test"}
    if name not in valid_names:
        raise ValueError(f"name must be one of {sorted(valid_names)}")

    processed_dir = Path(processed_dir)
    features = load_npz(processed_dir / f"features_{name}.npz")
    labels = np.load(processed_dir / f"labels_{name}.npy")
    return features, labels


def _validate_query_label_columns(df: pd.DataFrame, source: Path | str) -> pd.DataFrame:
    missing = {QUERY_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required column(s): {missing_list}")
    return df
