import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate)
from xgboost import XGBClassifier

from src.config import TRAIN_PATH, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from src.preprocessing import build_pipeline
from src.utils import load_xy

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc_ovr_weighted",
}

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss"),
}


PARAM_SPACES = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [2, 3, 4, 6],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
    },
}


def make_search(name, pipe, cv):
    if name == "Logistic Regression":
        return GridSearchCV(
            pipe,
            PARAM_SPACES[name],
            scoring=SCORING,
            refit="f1",
            cv=cv,
            n_jobs=-1
        )

    return RandomizedSearchCV(
        pipe,
        PARAM_SPACES[name],
        n_iter=20,
        scoring=SCORING,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X_train, y_train = load_xy(TRAIN_PATH)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    #  comparaison des 3 modèles

    rows = []
    for name, model in MODELS.items():
        res = cross_validate(build_pipeline(model), X_train, y_train, cv=cv,
                             scoring=SCORING, n_jobs=-1)
        row = {"model": name, "stage": "baseline"}
        row.update({m: res[f"test_{m}"].mean() for m in SCORING})
        rows.append(row)
        print(f"[baseline] {name}: F1={row['f1']:.3f}")

    baseline = pd.DataFrame(rows).sort_values("f1", ascending=False)

    # tuning des 2 meilleurs

    best_search, best_name, best_score = None, None, -1

    for name in baseline["model"].head(2):

        search = make_search(name, build_pipeline(MODELS[name]), cv)
        search.fit(X_train, y_train)

        print(f"[tuned] {name}: "
            f"F1={search.best_score_:.3f} | "
            f"{search.best_params_}")

        best_idx = search.best_index_

        row = {"model": name,"stage": "tuned"}

        row.update({
            m: search.cv_results_[f"mean_test_{m}"][best_idx]
            for m in SCORING
        })

        row["best_params"] = json.dumps(search.best_params_,default=str)

        rows.append(row)


        if search.best_score_ > best_score:
            best_search = search
            best_name = name
            best_score = search.best_score_

    # sauvegarde

    pd.DataFrame(rows).to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    joblib.dump(best_search.best_estimator_, MODELS_DIR / "best_pipeline.joblib")
    with open(REPORTS_DIR / "best_model.json", "w") as f:
        json.dump({"model": best_name, "cv_f1_weighted": best_score,
                   "params": best_search.best_params_}, f, indent=2, default=str)
    print(f"Meilleur modèle : {best_name} (F1 CV = {best_score:.3f})")


