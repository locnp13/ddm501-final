"""Data-quality tests for the ingest stage."""
import pandas as pd
import pandera as pa
import pytest

from src.training.data import clean, coerce_total_charges, validate


def make_row(**overrides) -> dict:
    """One valid raw Telco row."""
    row = {
        "customerID": "0001-A",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "149.25",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_valid_frame_passes_and_cleans() -> None:
    df = validate(coerce_total_charges(frame(make_row(), make_row(customerID="0002-B", Churn="Yes"))))
    out = clean(df)
    assert "customerID" not in out.columns
    assert out["Churn"].tolist() == [0, 1]


def test_blank_total_charges_becomes_nan_and_still_valid() -> None:
    df = coerce_total_charges(frame(make_row(TotalCharges=" ", tenure=0)))
    assert df["TotalCharges"].isna().all()
    validate(df)


def test_missing_column_fails() -> None:
    df = coerce_total_charges(frame(make_row())).drop(columns=["Contract"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_unknown_churn_label_fails() -> None:
    df = coerce_total_charges(frame(make_row(Churn="Maybe")))
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_duplicate_customer_id_fails() -> None:
    df = coerce_total_charges(frame(make_row(), make_row()))
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)
