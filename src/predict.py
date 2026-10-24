"""Predict churn for a single customer given a dict of raw inputs."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config
from src.features import transform

OPTIONAL = {"TotalCharges"}  # estimated as tenure * MonthlyCharges when omitted

# Retention playbook keyed by the engineered/one-hot feature that drives risk.
ACTIONS = {
    "Contract_Month-to-month": "Offer a discounted 1- or 2-year contract (month-to-month customers churn most).",
    "tenure": "Run an early-life check-in and onboarding offer; the first months are the riskiest.",
    "OnlineSecurity_No": "Offer a free trial of Online Security.",
    "TechSupport_No": "Offer free Tech Support for 3 months.",
    "InternetService_Fiber optic": "Check fiber service quality and price-match against competitors.",
    "PaymentMethod_Electronic check": "Incentivise switching to automatic payment (e.g. one-time bill credit).",
    "MonthlyCharges": "Review the plan; offer a right-sized bundle or loyalty discount.",
    "avg_charge_per_month": "Review the plan; offer a right-sized bundle or loyalty discount.",
    "PaperlessBilling": "Send a proactive value summary alongside the e-bill.",
    "OnlineBackup_No": "Bundle Online Backup into a discounted add-on package.",
    "SeniorCitizen": "Offer a senior-friendly plan and assisted support line.",
    "StreamingMovies_Yes": "Offer a streaming bundle discount.",
    "StreamingTV_Yes": "Offer a streaming bundle discount.",
}


@lru_cache(maxsize=1)
def load_model():
    """Load the fitted pipeline once."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run `python -m src.train` first.")
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold():
    """Tuned decision threshold saved by training."""
    return float(json.loads(config.METRICS_PATH.read_text())["threshold"])


def validate(inputs):
    """Check a dict of raw inputs and return a one-row dataframe.

    Raises ValueError with a message listing every problem found.
    """
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a dict of feature name -> value")
    required = [c for c in config.RAW_FEATURES if c not in OPTIONAL]
    errors = [f"missing field: {c}" for c in required if c not in inputs]
    errors += [f"unknown field: {c}" for c in inputs if c not in config.RAW_FEATURES]
    for col, allowed in config.CATEGORIES.items():
        if col in inputs and inputs[col] not in allowed:
            errors.append(f"{col}={inputs[col]!r} is invalid; allowed: {allowed}")
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        if col not in inputs:
            continue
        val = inputs[col]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val != val:
            errors.append(f"{col}={val!r} must be a number")
        elif not lo <= val <= hi:
            errors.append(f"{col}={val} out of range [{lo}, {hi}]")
    if errors:
        raise ValueError("Invalid input: " + "; ".join(errors))
    row = dict(inputs)
    if "TotalCharges" not in row:
        row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
    return pd.DataFrame([row])[config.RAW_FEATURES]


def risk_label(probability, threshold=None):
    """low: below the tuned threshold; medium: above it; high: at least 50%."""
    threshold = load_threshold() if threshold is None else threshold
    if probability >= 0.5:
        return "high"
    return "medium" if probability >= threshold else "low"


def predict(inputs):
    """Return churn probability, risk label and threshold for one customer."""
    df = validate(inputs)
    prob = float(load_model().predict_proba(df)[0, 1])
    thr = load_threshold()
    return {"probability": prob, "risk": risk_label(prob, thr), "threshold": thr}


def explain(inputs, top_n=8):
    """Top SHAP drivers (log-odds) for one customer, largest effect first."""
    from src.evaluate import make_explainer  # lazy: pulls in shap/matplotlib

    pipe = load_model()
    feats = transform(pipe, validate(inputs))
    values = make_explainer(pipe)(feats).values[0]
    drivers = pd.DataFrame({
        "feature": feats.columns, "value": feats.iloc[0].to_numpy(), "shap": values,
    })
    order = drivers["shap"].abs().sort_values(ascending=False).index[:top_n]
    return drivers.loc[order].reset_index(drop=True)


def suggest_actions(drivers, max_actions=3):
    """Retention actions for the top risk-increasing drivers."""
    actions = []
    for _, d in drivers[drivers["shap"] > 0].iterrows():
        text = ACTIONS.get(d["feature"])
        if text and text not in actions:
            actions.append(text)
    return actions[:max_actions]
