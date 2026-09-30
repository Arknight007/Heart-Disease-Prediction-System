"""
Central settings for the Heart Disease Prediction project.

Every other file imports its paths and column lists from here, so if you
want to change something (for example the test-set size) you only change it once.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "heart_disease_uci.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # makes every run give the same result
TEST_SIZE = 0.20           # 20% of patients are held back for the final test
CV_FOLDS = 5               # 5-fold cross-validation on the training data

# A column is dropped if more than this share of its values are missing.
MISSING_DROP_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGET = "target"          # 1 = heart disease present, 0 = not present

# Columns that are never used as model inputs:
#   id      - just a row number
#   dataset - which hospital the record came from. Disease rates differ a lot
#             between hospitals, so the model would "cheat" by learning the
#             hospital instead of the patient's health. A new patient in the app
#             also has no such value.
NON_FEATURE_COLUMNS = ["id", "dataset"]

# Final feature lists AFTER cleaning (ca and thal are removed by the
# missing-value rule above - see src/data.py).
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plain-language names used in charts and in the Streamlit app.
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalch": "Maximum heart rate achieved",
    "exang": "Chest pain during exercise",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thalassemia test result",
}

# Allowed values for each categorical input (these match the dataset).
CATEGORY_OPTIONS = {
    "sex": ["Male", "Female"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": ["no", "yes"],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": ["no", "yes"],
    "slope": ["upsloping", "flat", "downsloping"],
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression (baseline)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
