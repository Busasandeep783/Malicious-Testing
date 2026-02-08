import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, NUMERICAL_FEATURE_COLS, URL_COL, TARGET_COL


def load_data(filepath=DATA_PATH):
    """Load the phishing URL dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Clean the dataset:
    - Handle missing values
    - Remove duplicates
    - Ensure correct dtypes
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Handle missing values in numerical columns
    for col in NUMERICAL_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # Handle missing URLs
    df = df.dropna(subset=[URL_COL]).reset_index(drop=True)

    # Handle missing target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    print(f"After cleaning: {df.shape[0]} rows")
    print(f"Target distribution: {df[TARGET_COL].value_counts().to_dict()}")

    return df


if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    print(f"\nDataset ready: {df.shape}")
