"""Prediction API tests (use the trained model in models/)."""
import pytest

from src.predict import explain, predict, suggest_actions, validate

CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0,
}


def test_predict_returns_probability():
    out = predict(CUSTOMER)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["risk"] in {"low", "medium", "high"}


def test_risky_customer_scores_higher_than_loyal_one():
    loyal = dict(CUSTOMER, tenure=60, Contract="Two year", PaymentMethod="Bank transfer (automatic)")
    assert predict(CUSTOMER)["probability"] > predict(loyal)["probability"]


def test_missing_field_raises_clear_error():
    bad = {k: v for k, v in CUSTOMER.items() if k != "Contract"}
    with pytest.raises(ValueError, match="missing field: Contract"):
        validate(bad)


def test_invalid_category_and_range_raise():
    with pytest.raises(ValueError, match="Contract='Weekly' is invalid"):
        validate(dict(CUSTOMER, Contract="Weekly"))
    with pytest.raises(ValueError, match="out of range"):
        validate(dict(CUSTOMER, tenure=-5))
    with pytest.raises(ValueError, match="must be a number"):
        validate(dict(CUSTOMER, MonthlyCharges="a lot"))
    with pytest.raises(ValueError, match="unknown field"):
        validate(dict(CUSTOMER, favourite_colour="blue"))


def test_explain_and_actions():
    drivers = explain(CUSTOMER, top_n=5)
    assert len(drivers) == 5
    assert suggest_actions(drivers)
