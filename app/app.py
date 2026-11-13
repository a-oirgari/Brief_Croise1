import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # pour que joblib retrouve src.preprocessing

from src.config import INPUT_COLUMNS, LABELS, MODELS_DIR, REPORTS_DIR, TEST_PATH  # noqa: E402
from src.utils import load_xy  # noqa: E402

# ------------------------------------------------------------------ Palette
NAVY = "#142D4E"
NAVY_SOFT = "#1D3B63"
TEAL = "#0D9488"
TEAL_DARK = "#0F766E"
LIGHT_TEAL = "#CCFBF1"
BACKGROUND = "#F4F7FB"
WHITE = "#FFFFFF"
SLATE = "#64748B"
RED = "#DC2626"
BORDER = "#E2E8F0"

RISK_COLORS = {"Low": TEAL, "Medium": NAVY, "High": RED}
RISK_FR = {"Low": "Faible", "Medium": "Modéré", "High": "Élevé"}

st.set_page_config(page_title="Student Wellbeing Analytics", layout="wide")

# ---------------------------------------------------------------------- CSS
# Ce bloc remplace le fichier .streamlit/config.toml : tout le thème est ici.
st.markdown(
    f"""
    <style>
    /* ---------- Base ---------- */
    [data-testid="stApp"] {{ background: {BACKGROUND}; color: {NAVY}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }}
    h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 600; letter-spacing: -0.01em; }}
    label, [data-testid="stWidgetLabel"] p {{ color: {NAVY}; font-weight: 500; }}

    /* ---------- Barre latérale ---------- */
    [data-testid="stSidebar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #E2E8F0; }}
    [data-testid="stSidebar"] hr {{ border-color: {NAVY_SOFT}; }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #94A3B8; }}

    /* Navigation : le radio devient une liste de liens */
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0.25rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        padding: 0.55rem 0.8rem; border-radius: 8px; width: 100%; cursor: pointer;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {NAVY_SOFT}; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: {TEAL}; font-weight: 600;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: {WHITE}; }}

    /* ---------- Cartes (conteneurs avec key="card_...") ---------- */
    [class*="st-key-card"] {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 2px rgba(20, 45, 78, 0.04);
    }}

    /* ---------- Textes ---------- */
    .page-title {{ color: {NAVY}; font-size: 1.9rem; font-weight: 700; margin: 0; }}
    .page-subtitle {{ color: {SLATE}; font-size: 0.98rem; margin: 0.2rem 0 1.6rem 0; }}
    .section-title {{ color: {NAVY}; font-size: 1.02rem; font-weight: 600; margin: 0.4rem 0 0.2rem 0; }}
    .section-hint {{ color: {SLATE}; font-size: 0.85rem; margin-bottom: 0.8rem; }}

    /* ---------- KPI ---------- */
    .kpi {{
        background: {WHITE}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 1rem 1.2rem; border-top: 3px solid {TEAL};
    }}
    .kpi-label {{ color: {SLATE}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ color: {NAVY}; font-size: 1.8rem; font-weight: 700; line-height: 1.3; }}

    /* ---------- Marque ---------- */
    .brand {{ color: {WHITE} !important; font-size: 1.15rem; font-weight: 700; line-height: 1.25; }}
    .brand-sub {{ color: #94A3B8 !important; font-size: 0.8rem; margin-bottom: 1.4rem; }}

    /* ---------- Encadré d'information ---------- */
    .note {{
        background: {LIGHT_TEAL}; border-left: 4px solid {TEAL}; color: {NAVY};
        padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.9rem; margin-bottom: 1rem;
    }}

    /* ---------- Résultat de prédiction ---------- */
    .result-label {{ color: {SLATE}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .result-value {{ font-size: 2.4rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.2rem; }}
    .result-conf {{ color: {SLATE}; font-size: 0.9rem; margin-bottom: 1.2rem; }}
    .bar-row {{ display: flex; align-items: center; margin-bottom: 0.6rem; font-size: 0.88rem; color: {NAVY}; }}
    .bar-name {{ width: 70px; }}
    .bar-track {{ flex: 1; background: #EEF2F7; border-radius: 6px; height: 10px; margin: 0 0.8rem; }}
    .bar-fill {{ height: 10px; border-radius: 6px; }}
    .bar-pct {{ width: 48px; text-align: right; color: {SLATE}; }}

    /* ---------- Boutons (teal) ---------- */
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        border-radius: 8px; font-weight: 600; padding: 0.55rem 1.4rem;
        border: 1px solid {TEAL}; color: {TEAL}; background: {WHITE};
    }}
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"],
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
        background: {TEAL}; border-color: {TEAL}; color: {WHITE};
    }}
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {{
        background: {TEAL_DARK}; border-color: {TEAL_DARK}; color: {WHITE};
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {TEAL_DARK}; color: {TEAL_DARK}; }}
    .stButton > button p, .stDownloadButton > button p,
    [data-testid="stFormSubmitButton"] > button p {{ color: inherit; }}

    /* ---------- Champs : focus teal au lieu de rouge ---------- */
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {{
        border-color: {TEAL} !important;
    }}

    /* ---------- Sliders : teal au lieu de rouge ---------- */
    [data-testid="stSlider"] div[role="slider"] {{ background-color: {TEAL} !important; }}
    [data-testid="stSliderThumbValue"], [data-testid="stSlider"] [data-testid="stSliderThumbValue"] p {{
        color: {TEAL} !important;
    }}
    /* La barre pleine est une couleur imposée par Streamlit : on la décale de rouge vers teal */
    [data-testid="stSlider"] div[style*="linear-gradient"] {{ filter: hue-rotate(172deg); }}

    /* ---------- Zone d'import ---------- */
    [data-testid="stFileUploaderDropzone"] {{
        background: {BACKGROUND}; border: 1px dashed {TEAL}; border-radius: 10px;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ Chargement
@st.cache_resource
def load_pipeline():
    return joblib.load(MODELS_DIR / "best_pipeline.joblib")


@st.cache_data
def load_reports():
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    best = json.loads((REPORTS_DIR / "best_model.json").read_text())
    cv_results = pd.read_csv(REPORTS_DIR / "cv_results.csv")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv", index_col=0).iloc[:, 0]
    return metrics, best, cv_results, importance


@st.cache_data
def load_confusion():
    """Recalcule la matrice de confusion sur le jeu de test avec le pipeline sauvegardé."""
    X_test, y_test = load_xy(TEST_PATH)
    y_pred = load_pipeline().predict(X_test)
    return confusion_matrix(y_test, y_pred, labels=[0, 1, 2])


try:
    pipe = load_pipeline()
    metrics, best, cv_results, importance = load_reports()
except FileNotFoundError as e:
    st.error(f"Fichier introuvable : {e.filename}. Lance d'abord l'entraînement et l'évaluation.")
    st.stop()


# ------------------------------------------------------------------ Helpers UI
def card(name):
    # border=False : la bordure est dessinée par le CSS (classe st-key-card_...)
    return st.container(border=False, key=f"card_{name}")


def page_header(title, subtitle):
    st.markdown(f'<p class="page-title">{title}</p><p class="page-subtitle">{subtitle}</p>',
                unsafe_allow_html=True)


def section(title, hint=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<div class="section-hint">{hint}</div>', unsafe_allow_html=True)


def kpi(label, value):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=SLATE, labelsize=9)


def plot_confusion(cm):
    cmap = LinearSegmentedColormap.from_list("teal", [WHITE, TEAL])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(3), [RISK_FR[l] for l in LABELS])
    ax.set_yticks(range(3), [RISK_FR[l] for l in LABELS])
    ax.set_xlabel("Classe prédite", color=SLATE, fontsize=10)
    ax.set_ylabel("Classe réelle", color=SLATE, fontsize=10)
    for i in range(3):
        for j in range(3):
            color = WHITE if cm[i, j] > cm.max() * 0.55 else NAVY
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color,
                    fontsize=13, fontweight="bold")
    ax.tick_params(colors=SLATE, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def plot_importance(series, top=10):
    s = series.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.barh(s.index.str.replace("_", " "), s.values, color=TEAL, height=0.6)
    ax.set_xlabel("Baisse du F1 quand la variable est mélangée", color=SLATE, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    return fig


def result_card(pred_idx, proba):
    label = LABELS[pred_idx]
    color = RISK_COLORS[label]
    bars = ""
    for i, lab in enumerate(LABELS):
        bars += (
            f'<div class="bar-row"><span class="bar-name">{RISK_FR[lab]}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{proba[i] * 100:.1f}%; background:{RISK_COLORS[lab]};"></div></div>'
            f'<span class="bar-pct">{proba[i] * 100:.0f} %</span></div>'
        )
    st.markdown(
        f'<div class="result-label">Niveau de risque estimé</div>'
        f'<div class="result-value" style="color:{color};">{RISK_FR[label]}</div>'
        f'<div class="result-conf">Probabilité associée : {proba[pred_idx] * 100:.0f} %</div>'
        f"{bars}",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------ Barre latérale
with st.sidebar:
    st.markdown(
        '<div class="brand">Student Wellbeing<br>Analytics</div>'
        '<div class="brand-sub">Analyse du risque de santé mentale</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", ["Dashboard", "Single Prediction", "Batch Prediction"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Modèle actif : {best['model']}")


# ================================================================== DASHBOARD
if page == "Dashboard":
    page_header("Dashboard", f"Performances du modèle retenu ({best['model']}) sur le jeu de test.")

    kpis = [
        ("Accuracy", "accuracy"), ("Précision", "precision_weighted"),
        ("Rappel", "recall_weighted"), ("F1 pondéré", "f1_weighted"),
        ("ROC-AUC", "roc_auc_ovr"),
    ]
    for col, (label, key) in zip(st.columns(5), kpis):
        with col:
            kpi(label, f"{metrics[key]:.3f}")

    st.write("")
    left, right = st.columns(2)
    with left:
        with card("confusion"):
            section("Matrice de confusion", "Jeu de test, valeurs absolues.")
            st.pyplot(plot_confusion(load_confusion()), width="stretch")
    with right:
        with card("importance"):
            section("Importance des variables", "Les 10 variables les plus influentes (permutation).")
            st.pyplot(plot_importance(importance), width="stretch")

    st.write("")
    with card("report"):
        section("Rapport par classe", "Précision, rappel et F1 pour chaque niveau de risque.")
        report = pd.DataFrame(metrics["report"]).T.loc[LABELS].copy()
        report.index = [RISK_FR[l] for l in report.index]
        report = report.rename(columns={"precision": "Précision", "recall": "Rappel",
                                        "f1-score": "F1", "support": "Effectif"})
        st.dataframe(report.round(3), width="stretch")

    st.write("")
    with card("comparison"):
        section("Comparaison des modèles", "Validation croisée 5-fold sur le jeu d'entraînement.")
        cols = ["model", "stage", "accuracy", "precision", "recall", "f1", "roc_auc"]
        table = cv_results[[c for c in cols if c in cv_results.columns]].copy()
        table = table.rename(columns={
            "model": "Modèle", "stage": "Étape", "accuracy": "Accuracy",
            "precision": "Précision", "recall": "Rappel", "f1": "F1", "roc_auc": "ROC-AUC",
        })
        table["Étape"] = table["Étape"].map({"baseline": "Par défaut", "tuned": "Optimisé"})
        st.dataframe(table.round(3), width="stretch", hide_index=True)


# ========================================================= SINGLE PREDICTION
elif page == "Single Prediction":
    page_header("Single Prediction", "Saisir le profil d'un étudiant pour obtenir une estimation.")

    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        with card("form"):
            with st.form("student_form", border=False):
                section("Profil")
                c1, c2, c3 = st.columns(3)
                age = c1.number_input("Âge", 16, 40, 21)
                gender = c2.selectbox("Genre", ["Male", "Female", "Non-binary"],
                                      format_func={"Male": "Homme", "Female": "Femme",
                                                   "Non-binary": "Non-binaire"}.get)
                year = c3.number_input("Année d'étude", 1, 5, 2)

                section("Mode de vie")
                c1, c2 = st.columns(2)
                sleep = c1.slider("Sommeil (h / nuit)", 2.0, 12.0, 7.0, 0.5)
                activity = c2.selectbox("Activité physique", ["Low", "Moderate", "High"],
                                        format_func={"Low": "Faible", "Moderate": "Modérée",
                                                     "High": "Élevée"}.get)
                social = c1.slider("Réseaux sociaux (h / jour)", 0.0, 16.0, 3.0, 0.5)
                screen = c2.slider("Temps d'écran total (h / jour)", 0.0, 18.0, 6.0, 0.5)
                diet = c1.selectbox("Qualité de l'alimentation", ["Poor", "Average", "Good"],
                                    format_func={"Poor": "Mauvaise", "Average": "Moyenne",
                                                 "Good": "Bonne"}.get)

                section("Contexte académique et social")
                c1, c2 = st.columns(2)
                study = c1.slider("Étude (h / semaine)", 0.0, 80.0, 20.0, 1.0)
                gpa = c2.slider("GPA (sur 4)", 0.0, 4.0, 3.0, 0.1)
                att = c1.slider("Assiduité (%)", 0.0, 100.0, 80.0, 1.0)
                fin = c2.slider("Stress financier (1 à 10)", 1.0, 10.0, 5.0, 0.5)
                fam = c1.slider("Soutien familial (1 à 10)", 1.0, 10.0, 6.0, 0.5)
                job = c2.selectbox("Emploi à temps partiel", ["No", "Yes"],
                                   format_func={"No": "Non", "Yes": "Oui"}.get)
                hist = c1.selectbox("Antécédents de santé mentale", ["No", "Yes"],
                                    format_func={"No": "Non", "Yes": "Oui"}.get)

                submitted = st.form_submit_button("Lancer la prédiction", type="primary")

    with result_col:
        with card("result"):
            if submitted:
                row = pd.DataFrame([{
                    "Age": age, "Year_of_Study": year, "Sleep_Hours": sleep,
                    "Study_Hours_Per_Week": study, "GPA": gpa,
                    "Social_Media_Hours": social, "Screen_Time_Hours": screen,
                    "Financial_Stress": fin, "Family_Support": fam,
                    "Attendance_Rate": att, "Physical_Activity": activity,
                    "Part_Time_Job": job, "Diet_Quality": diet,
                    "Previous_Mental_Health_History": hist, "Gender": gender,
                }])[INPUT_COLUMNS]
                proba = pipe.predict_proba(row)[0]
                result_card(int(proba.argmax()), proba)
            else:
                section("Résultat", "Renseigne le formulaire puis lance la prédiction.")
                st.markdown('<div class="note">Le résultat s\'affichera ici.</div>',
                            unsafe_allow_html=True)


# =========================================================== BATCH PREDICTION
else:
    page_header("Batch Prediction", "Importer un fichier CSV et télécharger les prédictions.")

    left, right = st.columns([3, 2], gap="large")

    with right:
        with card("template"):
            section("Format attendu", "Le fichier doit contenir ces colonnes (l'ordre n'importe pas).")
            st.markdown(
                '<div class="note">' + ", ".join(INPUT_COLUMNS) + "</div>",
                unsafe_allow_html=True,
            )
            example = pd.DataFrame([{
                "Age": 21, "Year_of_Study": 2, "Sleep_Hours": 7.0, "Study_Hours_Per_Week": 20,
                "GPA": 3.0, "Social_Media_Hours": 3.0, "Screen_Time_Hours": 6.0,
                "Financial_Stress": 5.0, "Family_Support": 6.0, "Attendance_Rate": 80,
                "Physical_Activity": "Moderate", "Part_Time_Job": "No", "Diet_Quality": "Average",
                "Previous_Mental_Health_History": "No", "Gender": "Female",
            }])[INPUT_COLUMNS]
            st.download_button("Télécharger un modèle de fichier",
                               example.to_csv(index=False), "template_students.csv", "text/csv")

    with left:
        with card("upload"):
            section("Fichier à analyser")
            file = st.file_uploader("Fichier CSV", type="csv", label_visibility="collapsed")

    if file:
        data = pd.read_csv(file)
        missing = [c for c in INPUT_COLUMNS if c not in data.columns]

        st.write("")
        if missing:
            st.error(f"Colonnes manquantes : {', '.join(missing)}")
        else:
            with card("preview"):
                section(f"Aperçu ({len(data)} lignes)")
                st.dataframe(data.head(10), width="stretch", hide_index=True)

            if st.button("Lancer la prédiction", type="primary"):
                proba = pipe.predict_proba(data[INPUT_COLUMNS])
                out = data.copy()
                out["Risque_prédit"] = [RISK_FR[LABELS[i]] for i in proba.argmax(axis=1)]
                for i, lab in enumerate(LABELS):
                    out[f"Proba_{RISK_FR[lab]}"] = proba[:, i].round(3)
                st.session_state["batch_result"] = out

    if "batch_result" in st.session_state and file:
        out = st.session_state["batch_result"]
        st.write("")
        counts = out["Risque_prédit"].value_counts()
        for col, lab in zip(st.columns(3), LABELS):
            with col:
                kpi(f"Risque {RISK_FR[lab].lower()}", int(counts.get(RISK_FR[lab], 0)))

        st.write("")
        with card("batch_out"):
            section("Résultats")
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button("Télécharger les résultats", out.to_csv(index=False),
                               "predictions.csv", "text/csv", type="primary")