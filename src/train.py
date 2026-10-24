"""Train, compare and tune churn models; pick a threshold; save artifacts.

Run with:  python -m src.train
"""
import json
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from xgboost import XGBClassifier

from src import config, data
from src.features import FeatureEngineer, build_preprocessor

SCORING = {
    "pr_auc": "average_precision", "roc_auc": "roc_auc",
    "recall": "recall", "precision": "precision", "f1": "f1",
}
STRATEGIES = ["none", "class_weight", "smote"]


def make_model(name, strategy, pos_weight):
    """Build a classifier; the `class_weight` strategy uses its own weighting."""
    w = strategy == "class_weight"
    s = config.SEED
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=s,
                                  class_weight="balanced" if w else None)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=1,
            class_weight="balanced" if w else None)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=s, n_jobs=1, eval_metric="logloss",
            scale_pos_weight=pos_weight if w else 1.0)
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=s, n_jobs=1,
            verbose=-1, scale_pos_weight=pos_weight if w else 1.0)
    raise ValueError(name)


def make_pipeline(name, strategy, pos_weight):
    """Full pipeline: feature engineering -> preprocessing -> [SMOTE] -> model.

    SMOTE is a pipeline step, so in CV it only ever sees the training folds.
    Only logistic regression gets scaled numerics.
    """
    steps = [
        ("engineer", FeatureEngineer()),
        ("prep", build_preprocessor(scale=name == "Logistic Regression")),
    ]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=config.SEED)))
    steps.append(("clf", make_model(name, strategy, pos_weight)))
    return Pipeline(steps)


def metrics_at(y, proba, threshold=0.5):
    """Threshold-free (PR/ROC-AUC) and thresholded (P/R/F1) metrics."""
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred),
    }


def cost_curve(y, proba, thresholds=None):
    """Average cost per customer at each threshold (FN costs more than FP)."""
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    costs = []
    for t in thresholds:
        pred = proba >= t
        fn = np.sum((~pred) & (y == 1))
        fp = np.sum(pred & (y == 0))
        costs.append((config.COST_FN * fn + config.COST_FP * fp) / len(y))
    return thresholds, np.array(costs)


def tune(name, strategy, X, y, cv, pos_weight):
    """Light RandomizedSearchCV on PR-AUC for a boosted model."""
    if name == "XGBoost":
        space = {
            "clf__n_estimators": randint(150, 600),
            "clf__max_depth": randint(2, 6),
            "clf__learning_rate": loguniform(0.01, 0.2),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_weight": randint(1, 10),
            "clf__reg_lambda": loguniform(0.1, 20),
        }
    else:
        space = {
            "clf__n_estimators": randint(150, 600),
            "clf__num_leaves": randint(4, 40),
            "clf__learning_rate": loguniform(0.01, 0.2),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_samples": randint(10, 80),
            "clf__reg_lambda": loguniform(0.1, 20),
        }
    search = RandomizedSearchCV(
        make_pipeline(name, strategy, pos_weight), space,
        n_iter=config.N_SEARCH_ITER, scoring="average_precision", cv=cv,
        random_state=config.SEED, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search


def _row(name, strategy, cvres, test):
    row = {"model": name, "strategy": strategy}
    row.update({f"cv_{k}": float(cvres[f"test_{k}"].mean()) for k in SCORING})
    row.update({f"test_{k}": float(v) for k, v in test.items()})
    return row


def main():
    """Run the full experiment and write the model and metrics to models/."""
    warnings.filterwarnings("ignore")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    X_train, X_test, y_train, y_test = data.train_test()
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    print(f"train={len(X_train)} test={len(X_test)} "
          f"churn rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    # 1) Compare 4 models x 3 imbalance strategies with 5-fold stratified CV.
    rows, fitted = [], {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
        for strat in STRATEGIES:
            t0 = time.time()
            pipe = make_pipeline(name, strat, pos_weight)
            cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
            pipe.fit(X_train, y_train)
            test = metrics_at(y_test, pipe.predict_proba(X_test)[:, 1])
            rows.append(_row(name, strat, cvres, test))
            fitted[(name, strat)] = pipe
            print(f"{name:20s} {strat:13s} cv PR-AUC={rows[-1]['cv_pr_auc']:.4f} "
                  f"recall={rows[-1]['cv_recall']:.3f} ({time.time() - t0:.0f}s)")

    # 2) Tune the two boosted models with their best imbalance strategy.
    table = pd.DataFrame(rows)
    best_params = {}
    for name in ["XGBoost", "LightGBM"]:
        sub = table[table.model == name]
        strat = sub.loc[sub.cv_pr_auc.idxmax(), "strategy"]
        search = tune(name, strat, X_train, y_train, cv, pos_weight)
        best_params[name] = {k: (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}
        est = search.best_estimator_
        cvres = cross_validate(est, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        test = metrics_at(y_test, est.predict_proba(X_test)[:, 1])
        rows.append(_row(f"{name} (tuned)", strat, cvres, test))
        fitted[(f"{name} (tuned)", strat)] = est
        print(f"{name} tuned ({strat}): cv PR-AUC={rows[-1]['cv_pr_auc']:.4f}")

    # 3) Choose the final model by CV PR-AUC only (test set is never used to choose).
    table = pd.DataFrame(rows)
    best = table.loc[table.cv_pr_auc.idxmax()]
    final = fitted[(best.model, best.strategy)]
    print(f"\nSelected: {best.model} / {best.strategy}")

    # 4) Tune the decision threshold on out-of-fold train predictions.
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba",
                            n_jobs=-1)[:, 1]
    thr, costs = cost_curve(y_train, oof)
    threshold = float(thr[np.argmin(costs)])
    proba_test = final.predict_proba(X_test)[:, 1]

    def test_cost(t):
        return float(cost_curve(y_test, proba_test, np.array([t]))[1][0])

    print(f"threshold={threshold:.2f}  oof cost={costs.min():.3f}  "
          f"test cost={test_cost(threshold):.3f} (at 0.5: {test_cost(0.5):.3f})")

    out = {
        "seed": config.SEED,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "churn_rate": float(np.mean(np.r_[y_train, y_test])),
        "selected": {"model": best.model, "strategy": best.strategy},
        "threshold": threshold,
        "cost_assumption": {"false_negative": config.COST_FN,
                            "false_positive": config.COST_FP},
        "test_metrics_at_0.5": metrics_at(y_test, proba_test, 0.5),
        "test_metrics_at_threshold": metrics_at(y_test, proba_test, threshold),
        "cost_per_customer": {
            "oof_train_at_threshold": float(costs.min()),
            "test_at_threshold": test_cost(threshold),
            "test_at_0.5": test_cost(0.5),
            "test_flag_nobody": float(config.COST_FN * y_test.mean()),
            "test_flag_everybody": float(config.COST_FP * (1 - y_test.mean())),
        },
        "cost_curve": {"thresholds": thr.tolist(), "costs": costs.tolist()},
        "best_params": best_params,
        "comparison": rows,
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(out, indent=2))
    print("saved", config.MODEL_PATH, config.METRICS_PATH)


if __name__ == "__main__":
    main()
