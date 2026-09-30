"""
Loading, checking and cleaning the UCI Heart Disease data.

Steps performed here:
1. Load the CSV (or read it back from the database).
2. Report missing values for every column.
3. Fix values that are "hidden" missing values (cholesterol or blood pressure of 0).
4. Drop columns that are mostly empty.
5. Turn the 0-4 diagnosis column into a simple yes/no target.
"""
import numpy as np
import pandas as pd

from . import config


def load_raw_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the raw UCI Heart Disease CSV file."""
    return pd.read_csv(path)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table showing how many values are missing in each column."""
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(1),
    })
    return report.sort_values("missing_percent", ascending=False)


def _to_yes_no(series: pd.Series) -> pd.Series:
    """Convert True/False style values into 'yes'/'no' text (keeps missing as NaN)."""
    mapping = {True: "yes", False: "no", "TRUE": "yes", "FALSE": "no",
               "True": "yes", "False": "no"}
    return series.map(mapping)


def clean_data(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply all cleaning rules and return a model-ready table (features + target)."""
    df = raw.copy()

    # 1) A cholesterol or resting blood pressure of 0 is impossible for a living
    #    patient - these are really missing values that were written as 0.
    for col in ["chol", "trestbps"]:
        zero_count = int((df[col] == 0).sum())
        df.loc[df[col] == 0, col] = np.nan
        if verbose and zero_count:
            print(f"  - {col}: {zero_count} zero values treated as missing")

    # 2) Make the yes/no columns consistent text values.
    for col in ["fbs", "exang"]:
        df[col] = _to_yes_no(df[col])

    # 3) Drop columns where more than half the values are missing. Filling in
    #    that much data would mostly be guessing.
    candidate_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLUMNS + ["num"]]
    share_missing = df[candidate_cols].isna().mean()
    to_drop = share_missing[share_missing > config.MISSING_DROP_THRESHOLD].index.tolist()
    if verbose:
        for col in to_drop:
            print(f"  - dropping '{col}' ({share_missing[col]:.0%} missing)")
    df = df.drop(columns=to_drop)

    # 4) Target: the original 'num' column is 0 (no disease) to 4 (severe).
    #    We predict presence of disease, so anything above 0 becomes 1.
    df[config.TARGET] = (df["num"] > 0).astype(int)

    # 5) Keep only the model inputs and the target.
    df = df[config.ALL_FEATURES + [config.TARGET]]

    # 6) Drop exact duplicate rows, if any.
    before = len(df)
    df = df.drop_duplicates()
    if verbose and before != len(df):
        print(f"  - removed {before - len(df)} duplicate rows")

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """Split a cleaned table into inputs (X) and the answer column (y)."""
    return df[config.ALL_FEATURES], df[config.TARGET]
