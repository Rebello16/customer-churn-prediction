"""Feature engineering and the preprocessing ColumnTransformer.

Everything is stateless or lives inside a sklearn Pipeline, so statistics such
as scaling means or category lists are learned from training folds only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]
ENGINEERED_NUMERIC = ["n_services", "avg_charge_per_month", "has_support", "autopay"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered columns to a raw (cleaned) customer dataframe.

    It learns nothing from the data, so it cannot leak information.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_bucket"] = pd.cut(
            X["tenure"], TENURE_BINS, labels=TENURE_LABELS
        ).astype(str)
        # Subscribed add-on services ("Yes"); includes phone lines and streaming.
        X["n_services"] = (X[config.SERVICES] == "Yes").sum(axis=1)
        # New customers (tenure 0) have no history, so use their monthly charge.
        X["avg_charge_per_month"] = np.where(
            X["tenure"] > 0,
            X["TotalCharges"] / X["tenure"].clip(lower=1),
            X["MonthlyCharges"],
        )
        X["has_support"] = (X[config.SUPPORT_SERVICES] == "Yes").any(axis=1).astype(int)
        X["autopay"] = X["PaymentMethod"].isin(config.AUTOPAY_METHODS).astype(int)
        for col in config.BINARY:
            X[col] = X[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
        return X


def build_preprocessor(scale=False):
    """ColumnTransformer: one-hot nominal columns, optionally scale numerics.

    Tree models do not need scaling; logistic regression does.
    """
    numeric = config.NUMERIC + ENGINEERED_NUMERIC
    passthrough = config.BINARY + ["SeniorCitizen"]
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("bin", "passthrough", passthrough),
            ("cat", onehot, config.NOMINAL + ["tenure_bucket"]),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_features(scale=False):
    """Feature engineering + preprocessing as one Pipeline (no model)."""
    return Pipeline(
        [("engineer", FeatureEngineer()), ("prep", build_preprocessor(scale))]
    )
