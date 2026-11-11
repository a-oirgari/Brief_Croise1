import joblib

from src.config import TRAIN_PATH, MODELS_DIR
from src.preprocessing import build_feature_prep
from src.utils import load_xy


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X_train, _ = load_xy(TRAIN_PATH)

    prep = build_feature_prep()
    X_t = prep.fit_transform(X_train)
    joblib.dump(prep, MODELS_DIR / "preprocessor.joblib")
    print("Préprocesseur sauvegardé. Shape transformée :", X_t.shape)


