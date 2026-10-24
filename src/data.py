"""Data download, loading and cleaning."""
import urllib.request

import numpy as np
import pandas as pd

from src import config

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download(path=config.RAW_PATH):
    """Download the raw dataset if it is not on disk yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path


def clean(df):
    """Fix dataset quirks.

    * TotalCharges is text with blanks for tenure-0 customers (new, not billed
      yet). Their total is genuinely 0, so we fill with 0.
    * customerID carries no signal and is dropped.
    * Churn "Yes"/"No" becomes 1/0 when present.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0.0
    df = df.drop(columns=["customerID"], errors="ignore")
    if config.TARGET in df.columns:
        df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df


def load_raw(path=config.RAW_PATH):
    """Read the raw CSV, downloading it first if needed."""
    return pd.read_csv(download(path))


def load_clean(path=config.RAW_PATH):
    """Return the cleaned dataframe (features + 0/1 Churn)."""
    df = clean(load_raw(path))
    assert not df.isna().any().any(), "unexpected missing values after cleaning"
    return df


def split_xy(df):
    """Split a cleaned dataframe into features and target."""
    return df.drop(columns=[config.TARGET]), df[config.TARGET].to_numpy(dtype=np.int64)
