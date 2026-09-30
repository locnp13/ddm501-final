"""Tests for the feature-engineering transformer."""
import numpy as np
import pandas as pd

from src.training.features import FeatureEngineer
from tests.test_data import frame, make_row


def test_derived_features() -> None:
    row = make_row(
        tenure=10, TotalCharges="100", OnlineSecurity="Yes", OnlineBackup="Yes", InternetService="Fiber optic",
        TechSupport="No", PaymentMethod="Credit card (automatic)",
    )
    out = FeatureEngineer().transform(frame(row)).iloc[0]
    assert out["tenure_bucket"] == "7-12"
    assert out["num_services"] == 2
    assert out["avg_charge_per_month"] == 10.0
    assert out["autopay"] == 1
    assert out["has_internet_no_support"] == 0  # has online security


def test_new_customer_and_blank_total_charges() -> None:
    out = FeatureEngineer().transform(frame(make_row(tenure=0, TotalCharges=" "))).iloc[0]
    assert np.isnan(out["TotalCharges"])
    assert np.isnan(out["avg_charge_per_month"])  # tenure 0: left for the imputer
    assert out["tenure_bucket"] == "0-6"


def test_none_total_charges_from_api() -> None:
    out = FeatureEngineer().transform(pd.DataFrame([{**make_row(), "TotalCharges": None}]))
    assert np.isnan(out.loc[0, "TotalCharges"])


def test_drop_sensitive_keeps_raw_input_contract() -> None:
    df = frame(make_row())
    out = FeatureEngineer(add_features=False, drop_sensitive=True).transform(df)
    assert "gender" not in out.columns and "SeniorCitizen" not in out.columns
    assert "gender" in df.columns  # input untouched
    assert "num_services" not in out.columns
