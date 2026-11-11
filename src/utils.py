import pandas as pd
from src.config import TARGET, LABEL_TO_INT


def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map(LABEL_TO_INT)
    return X, y
