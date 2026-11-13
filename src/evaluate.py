import json
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score)

from src.config import TEST_PATH, MODELS_DIR, REPORTS_DIR, LABELS, RANDOM_STATE
from src.utils import load_xy


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    pipe = joblib.load(MODELS_DIR / "best_pipeline.joblib")
    X_test, y_test = load_xy(TEST_PATH)

    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"),
        "report": classification_report(y_test, y_pred, target_names=LABELS, output_dict=True),
    }


    wrong = y_test.values != y_pred
    metrics["n_errors"] = int(wrong.sum())

    metrics["n_severe_errors"] = int(((y_test.values - y_pred) ** 2 == 4).sum())

    errors = X_test[wrong].copy()
    errors["true"] = [LABELS[i] for i in y_test.values[wrong]]
    errors["pred"] = [LABELS[i] for i in y_pred[wrong]]
    errors.to_csv(REPORTS_DIR / "errors.csv", index=False)


    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)})

