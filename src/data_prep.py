import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (RAW_PATH, PROCESSED_DIR, TRAIN_PATH, TEST_PATH,
    TARGET, VALID_RANGES, RANDOM_STATE)


def clean(df):
    df = df.drop(columns=["Student_ID"], errors="ignore")

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    for col, (low, high) in VALID_RANGES.items():
        invalid = df[col].notna() & ~df[col].between(low, high)
        df.loc[invalid, col] = np.nan
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(pd.read_csv(RAW_PATH))

    train, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE)

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Train : {train.shape} | Test : {test.shape}")


