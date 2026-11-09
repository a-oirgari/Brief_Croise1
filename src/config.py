from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "students.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TARGET = "Mental_Health_Risk"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}

NUMERIC_COLS = [
    "Age", "Year_of_Study", "Sleep_Hours", "Study_Hours_Per_Week", "GPA",
    "Social_Media_Hours", "Screen_Time_Hours", "Financial_Stress",
    "Family_Support", "Attendance_Rate",
]
ENGINEERED_COLS = ["Sleep_Deficit", "Non_Social_Screen_Time", "Stress_Support_Gap"]


ORDINAL_COLS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Part_Time_Job": ["No", "Yes"],
    "Previous_Mental_Health_History": ["No", "Yes"],
}
NOMINAL_COLS = ["Gender"]

INPUT_COLUMNS = NUMERIC_COLS + list(ORDINAL_COLS) + NOMINAL_COLS


VALID_RANGES = {
    "Age": (16, 40),
    "Year_of_Study": (1, 5),
    "Sleep_Hours": (2, 12),
    "Study_Hours_Per_Week": (0, 80),
    "GPA": (0, 4),
    "Social_Media_Hours": (0, 16),
    "Screen_Time_Hours": (0, 18),
    "Financial_Stress": (1, 10),
    "Family_Support": (1, 10),
    "Attendance_Rate": (0, 100),
}