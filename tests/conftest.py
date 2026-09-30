"""Shared fixtures."""
import numpy as np
import pandas as pd
import pytest

from src.training.data import clean, coerce_total_charges, validate


def synthetic_telco(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Random frame that satisfies the Telco schema, with churn linked to contract and tenure."""
    rng = np.random.default_rng(seed)
    yes_no, addon = ["Yes", "No"], ["Yes", "No", "No internet service"]
    tenure = rng.integers(0, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.2, 0.2])
    monthly = rng.uniform(20, 110, n).round(2)
    churn_p = 0.08 + 0.35 * (contract == "Month-to-month") * (tenure < 12)
    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n),
            "tenure": tenure,
            "PhoneService": rng.choice(yes_no, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            **{c: rng.choice(addon, n) for c in
               ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]},
            "Contract": contract,
            "PaperlessBilling": rng.choice(yes_no, n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": (monthly * tenure).round(2).astype(str),
            "Churn": np.where(rng.random(n) < churn_p, "Yes", "No"),
        }
    )
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    return df


@pytest.fixture
def telco_clean() -> pd.DataFrame:
    """Cleaned synthetic data as the train stage reads it (Churn as 0/1, no customerID)."""
    return clean(validate(coerce_total_charges(synthetic_telco())))
