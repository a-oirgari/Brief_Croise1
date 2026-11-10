import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler)

from src.config import (NUMERIC_COLS, ENGINEERED_COLS, ORDINAL_COLS, NOMINAL_COLS)


def add_features(df):
    df = df.copy()
    df["Sleep_Deficit"] = (7 - df["Sleep_Hours"]).clip(lower=0)
    df["Non_Social_Screen_Time"] = (df["Screen_Time_Hours"] - df["Social_Media_Hours"]).clip(lower=0)
    df["Stress_Support_Gap"] = df["Financial_Stress"] - df["Family_Support"]
    return df


class IQRClipper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),
    ])
    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=list(ORDINAL_COLS.values()),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS + ENGINEERED_COLS),
        ("ord", ordinal, list(ORDINAL_COLS)),
        ("nom", nominal, NOMINAL_COLS),
    ])


def build_feature_prep():
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocessor", build_preprocessor()),
    ])


def build_pipeline(model):
    return Pipeline([
        ("prep", build_feature_prep()),
        ("model", model),
    ])