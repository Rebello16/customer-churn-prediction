"""Feature pipeline tests."""
import pytest

from src import config, data
from src.features import ENGINEERED_NUMERIC, build_features


@pytest.fixture(scope="module")
def df():
    return data.load_clean()


def test_no_nans_and_expected_columns(df):
    X, _ = data.split_xy(df)
    out = build_features().fit_transform(X)
    assert not out.isna().any().any()
    assert out.shape[0] == len(df)
    for col in config.NUMERIC + ENGINEERED_NUMERIC + ["Contract_Month-to-month"]:
        assert col in out.columns
    assert any(c.startswith("tenure_bucket_") for c in out.columns)


def test_scaling_flag(df):
    X, _ = data.split_xy(df)
    scaled = build_features(scale=True).fit_transform(X)
    assert abs(scaled["tenure"].mean()) < 1e-6
    assert scaled["tenure"].std(ddof=0) == pytest.approx(1, abs=1e-6)


def test_cleaning_handles_blank_total_charges():
    raw = data.load_raw()
    clean = data.clean(raw)
    assert clean["TotalCharges"].notna().all()
    assert (clean.loc[clean["tenure"] == 0, "TotalCharges"] == 0).all()
    assert "customerID" not in clean.columns
    assert set(clean[config.TARGET].unique()) == {0, 1}
