"""Central configuration: paths, seeds, column groups and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
N_SEARCH_ITER = 12

TARGET = "Churn"

# Business assumption: a missed churner costs 5x a wasted retention offer.
COST_FN = 5.0
COST_FP = 1.0

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
SERVICES = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
SUPPORT_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
NOMINAL = ["InternetService", "Contract", "PaymentMethod"] + SERVICES
AUTOPAY_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]

# Allowed values for each categorical input (used for validation in predict.py).
CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}
NUMERIC_RANGES = {
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 20000),
}
RAW_FEATURES = list(CATEGORIES) + list(NUMERIC_RANGES)
