"""Streamlit app: score a customer, see why, and review model performance."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, predict  # noqa: E402

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def get_model():
    """Load the pipeline once per server process."""
    return predict.load_model()


@st.cache_data
def get_metrics():
    """Load metrics.json written by training/evaluation."""
    return json.loads(config.METRICS_PATH.read_text())


def customer_form():
    """Sidebar form; returns a dict of raw inputs with sensible defaults."""
    c = config.CATEGORIES
    st.sidebar.header("Customer profile")
    d = {}
    d["gender"] = st.sidebar.selectbox("Gender", c["gender"])
    d["SeniorCitizen"] = int(st.sidebar.checkbox("Senior citizen"))
    d["Partner"] = st.sidebar.selectbox("Partner", c["Partner"], index=1)
    d["Dependents"] = st.sidebar.selectbox("Dependents", c["Dependents"], index=1)
    d["tenure"] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    d["Contract"] = st.sidebar.selectbox("Contract", c["Contract"])
    d["PaymentMethod"] = st.sidebar.selectbox("Payment method", c["PaymentMethod"])
    d["PaperlessBilling"] = st.sidebar.selectbox("Paperless billing", c["PaperlessBilling"])
    d["PhoneService"] = st.sidebar.selectbox("Phone service", c["PhoneService"])
    d["InternetService"] = st.sidebar.selectbox("Internet service", c["InternetService"], index=1)
    d["MonthlyCharges"] = st.sidebar.number_input(
        "Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)

    # Dependent fields: telephony/internet add-ons are forced to the "no service" value.
    no_internet = d["InternetService"] == "No"
    d["MultipleLines"] = (
        "No phone service" if d["PhoneService"] == "No"
        else st.sidebar.selectbox("Multiple lines", ["No", "Yes"]))
    for col in config.SERVICES[1:]:
        label = col.replace("Online", "Online ").replace("Tech", "Tech ")
        d[col] = ("No internet service" if no_internet
                  else st.sidebar.selectbox(label, ["No", "Yes"]))
    return d


def driver_chart(drivers):
    """Horizontal bar chart of SHAP drivers (red raises risk, blue lowers it)."""
    d = drivers.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(d["feature"], d["shap"], color=["#C44E52" if s > 0 else "#4C72B0" for s in d["shap"]])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Effect on churn risk (log-odds): red = raises, blue = lowers")
    fig.tight_layout()
    return fig


def predict_tab(inputs):
    """Probability, risk label, top reasons and suggested actions."""
    try:
        result = predict.predict(inputs)
        drivers = predict.explain(inputs, top_n=8)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return
    prob, risk, thr = result["probability"], result["risk"], result["threshold"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{prob:.1%}")
    col2.metric("Risk level", {"low": "🟢 Low", "medium": "🟠 Medium", "high": "🔴 High"}[risk])
    col3.metric("Decision threshold", f"{thr:.2f}")
    st.caption(
        f"Low: below the tuned threshold ({thr:.2f}). Medium: at or above it, where a "
        f"retention offer is worth its cost (missed churner = {config.COST_FN:g}x an offer). "
        "High: 50% or more.")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Top reasons")
        st.pyplot(driver_chart(drivers))
    with right:
        st.subheader("Suggested retention actions")
        actions = predict.suggest_actions(drivers) if risk != "low" else []
        if actions:
            for a in actions:
                st.markdown(f"- {a}")
        elif risk == "low":
            st.write("No action needed; customer is below the intervention threshold.")
        else:
            st.write("No specific driver-based action found; consider a general check-in.")


def performance_tab():
    """Model comparison table and evaluation plots."""
    try:
        m = get_metrics()
    except FileNotFoundError:
        st.error("metrics.json not found. Run `python -m src.train` and `python -m src.evaluate`.")
        return
    sel = m["selected"]
    t = m["test_metrics_at_threshold"]
    st.markdown(
        f"**Selected model:** {sel['model']} (imbalance strategy: {sel['strategy']}). "
        f"At threshold {m['threshold']:.2f} on the held-out test set: recall {t['recall']:.2f}, "
        f"precision {t['precision']:.2f}, F1 {t['f1']:.2f}, PR-AUC {t['pr_auc']:.3f}, "
        f"ROC-AUC {t['roc_auc']:.3f}.")
    df = pd.DataFrame(m["comparison"])
    cols = ["model", "strategy", "cv_pr_auc", "cv_roc_auc", "test_pr_auc", "test_roc_auc",
            "test_recall", "test_precision", "test_f1"]
    st.subheader("Model comparison")
    st.caption("CV = 5-fold stratified on the training set. Test = held-out 20%. "
               "Recall/precision/F1 here are at the default 0.5 threshold.")
    st.dataframe(df[cols].round(3), hide_index=True, use_container_width=True)
    figs = [("roc_curve.png", "ROC"), ("pr_curve.png", "Precision-recall"),
            ("confusion_matrix.png", "Confusion matrix"), ("calibration.png", "Calibration"),
            ("cost_curve.png", "Cost curve"), ("gain_lift.png", "Gain and lift"),
            ("shap_summary.png", "SHAP summary"), ("shap_single_customer.png", "SHAP single customer")]
    cols = st.columns(2)
    for i, (name, title) in enumerate(figs):
        path = config.FIGURES_DIR / name
        if path.exists():
            cols[i % 2].image(str(path), caption=title)


def main():
    """App entry point."""
    st.title("📉 Telco Customer Churn Predictor")
    try:
        get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    inputs = customer_form()
    tab1, tab2 = st.tabs(["Predict", "Model performance"])
    with tab1:
        predict_tab(inputs)
    with tab2:
        performance_tab()


main()
