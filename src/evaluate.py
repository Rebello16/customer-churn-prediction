"""Evaluate the saved pipeline on the held-out test set and draw all figures.

Run with:  python -m src.evaluate
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)

from src import config, data  # noqa: E402
from src.features import FeatureEngineer, transform  # noqa: E402

BLUE, RED, GREY = "#4C72B0", "#C44E52", "#8c8c8c"


def make_explainer(pipe):
    """SHAP TreeExplainer for the pipeline's (tree-based) classifier."""
    return shap.TreeExplainer(pipe.named_steps["clf"])


def shap_for(pipe, explainer, X):
    """Return (feature dataframe, SHAP values array) for raw customer rows X."""
    feats = transform(pipe, X)
    values = explainer(feats).values
    return feats, values


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y, proba, thr):
    """Confusion matrix at the tuned threshold."""
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, proba >= thr, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix (threshold = {thr:.2f})")
    _save(fig, "confusion_matrix.png")


def plot_roc_pr(y, proba):
    """ROC and precision-recall curves."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, color=BLUE)
    ax.plot([0, 1], [0, 1], "--", color=GREY)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(f"ROC curve (AUC = {roc_auc_score(y, proba):.3f})")
    _save(fig, "roc_curve.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    prec, rec, _ = precision_recall_curve(y, proba)
    ax.plot(rec, prec, color=BLUE)
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.axhline(y.mean(), ls="--", color=GREY, label=f"Baseline = {y.mean():.2f}")
    ax.legend()
    ax.set_title(f"Precision-recall curve (AP = {average_precision_score(y, proba):.3f})")
    _save(fig, "pr_curve.png")


def plot_calibration(y, proba):
    """Reliability diagram: predicted probability vs observed churn rate."""
    frac, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect")
    ax.plot(mean_pred, frac, "o-", color=BLUE, label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate",
           title="Calibration plot")
    ax.legend()
    _save(fig, "calibration.png")


def gain_lift(y, proba):
    """Decile table: cumulative gain and lift when ranking customers by risk."""
    df = pd.DataFrame({"y": y, "p": proba}).sort_values("p", ascending=False)
    df["decile"] = np.arange(len(df)) * 10 // len(df) + 1
    g = df.groupby("decile").y.agg(["sum", "size"])
    out = pd.DataFrame({
        "pct_customers": g["size"].cumsum() / len(df),
        "gain": g["sum"].cumsum() / df.y.sum(),
        "lift": (g["sum"] / g["size"]) / df.y.mean(),
    })
    return out


def plot_gain_lift(y, proba):
    """Cumulative gain chart and lift-by-decile chart."""
    t = gain_lift(y, proba)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(np.r_[0, t.pct_customers], np.r_[0, t.gain], "o-", color=BLUE, label="Model")
    ax[0].plot([0, 1], [0, 1], "--", color=GREY, label="Random")
    ax[0].set(xlabel="Share of customers contacted (highest risk first)",
              ylabel="Share of churners captured", title="Cumulative gain")
    ax[0].legend()
    ax[1].bar(t.index, t.lift, color=BLUE)
    ax[1].axhline(1, ls="--", color=GREY)
    ax[1].set(xlabel="Risk decile (1 = highest)", ylabel="Lift vs. average",
              title="Lift by decile")
    _save(fig, "gain_lift.png")
    return t


def plot_cost_curve(metrics):
    """Expected cost per customer across thresholds, with the chosen optimum."""
    thr = np.array(metrics["cost_curve"]["thresholds"])
    cost = np.array(metrics["cost_curve"]["costs"])
    best = metrics["threshold"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr, cost, color=BLUE)
    ax.axvline(best, color=RED, ls="--", label=f"Chosen threshold = {best:.2f}")
    ax.axvline(0.5, color=GREY, ls=":", label="Default 0.50")
    ax.set(xlabel="Decision threshold", ylabel="Cost per customer (offer = 1 unit)",
           title=f"Cost curve (missed churner = {config.COST_FN:g}x wasted offer)")
    ax.legend()
    _save(fig, "cost_curve.png")


def plot_shap(pipe, X_test, proba, y_test):
    """SHAP summary plot plus a waterfall for one high-risk churner."""
    explainer = make_explainer(pipe)
    feats = transform(pipe, X_test)
    exp = explainer(feats)
    shap.summary_plot(exp.values, feats, show=False, max_display=15)
    _save(plt.gcf(), "shap_summary.png")
    shap.summary_plot(exp.values, feats, plot_type="bar", show=False, max_display=15)
    _save(plt.gcf(), "shap_importance.png")
    # Example customer: an actual churner whose predicted risk is close to 0.7.
    idx = int(np.argmin(np.where(y_test == 1, np.abs(proba - 0.7), np.inf)))
    shap.plots.waterfall(exp[idx], max_display=12, show=False)
    _save(plt.gcf(), "shap_single_customer.png")
    return idx, exp


def segment_report(X, y, proba, thr):
    """Recall and misses by customer segment on the test set."""
    X = FeatureEngineer().transform(X)
    df = X.assign(y=y, pred=(proba >= thr).astype(int))
    rows = []
    for col in ["Contract", "tenure_bucket", "InternetService", "PaymentMethod", "SeniorCitizen"]:
        for val, g in df.groupby(col, observed=True):
            pos = g[g.y == 1]
            rows.append({
                "segment": f"{col}={val}", "customers": int(len(g)),
                "churn_rate": float(g.y.mean()), "churners": int(len(pos)),
                "recall": float(pos.pred.mean()) if len(pos) else None,
                "precision": float(g[g.pred == 1].y.mean()) if g.pred.sum() else None,
                "missed_churners": int(((g.pred == 0) & (g.y == 1)).sum()),
            })
    return rows


def main():
    """Draw every figure and add test-set analysis to metrics.json."""
    pipe = joblib.load(config.MODEL_PATH)
    metrics = json.loads(config.METRICS_PATH.read_text())
    thr = metrics["threshold"]
    _, X_test, _, y_test = data.train_test()
    proba = pipe.predict_proba(X_test)[:, 1]

    plot_confusion(y_test, proba, thr)
    plot_roc_pr(y_test, proba)
    plot_calibration(y_test, proba)
    deciles = plot_gain_lift(y_test, proba)
    plot_cost_curve(metrics)
    idx, _ = plot_shap(pipe, X_test, proba, y_test)

    pred = proba >= thr
    metrics["confusion_at_threshold"] = {
        "tn": int(((~pred) & (y_test == 0)).sum()), "fp": int((pred & (y_test == 0)).sum()),
        "fn": int(((~pred) & (y_test == 1)).sum()), "tp": int((pred & (y_test == 1)).sum()),
    }
    metrics["gain_lift_deciles"] = deciles.round(4).reset_index().to_dict("records")
    metrics["segments"] = segment_report(X_test, y_test, proba, thr)
    metrics["example_customer"] = {"test_row": idx, "probability": float(proba[idx])}
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("figures:", sorted(p.name for p in config.FIGURES_DIR.glob("*.png")))
    print(metrics["confusion_at_threshold"])
    print(deciles.round(3))
    print(pd.DataFrame(metrics["segments"]).round(3).to_string())


if __name__ == "__main__":
    main()
