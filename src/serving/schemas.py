"""Request/response schemas for the prediction API."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]

# Categories seen in training; keep in sync with the Pandera SCHEMA (tests/test_api.py checks this).
KNOWN_CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADDON,
    "OnlineBackup": ADDON,
    "DeviceProtection": ADDON,
    "TechSupport": ADDON,
    "StreamingTV": ADDON,
    "StreamingMovies": ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

EXAMPLE: dict[str, Any] = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 256.5,
}


def _category(description: str) -> Any:
    """Required non-empty text. Unknown values are accepted and reported as warnings, not rejected."""
    return Field(min_length=1, max_length=64, description=description)


class CustomerFeatures(BaseModel):
    """One customer, using the raw Telco columns. Feature engineering happens inside the model."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE]})

    gender: str = _category("Female or Male")
    SeniorCitizen: int = Field(ge=0, le=1, description="1 if the customer is a senior citizen")
    Partner: str = _category("Yes or No")
    Dependents: str = _category("Yes or No")
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: str = _category("Yes or No")
    MultipleLines: str = _category("Yes, No or No phone service")
    InternetService: str = _category("DSL, Fiber optic or No")
    OnlineSecurity: str = _category("Yes, No or No internet service")
    OnlineBackup: str = _category("Yes, No or No internet service")
    DeviceProtection: str = _category("Yes, No or No internet service")
    TechSupport: str = _category("Yes, No or No internet service")
    StreamingTV: str = _category("Yes, No or No internet service")
    StreamingMovies: str = _category("Yes, No or No internet service")
    Contract: str = _category("Month-to-month, One year or Two year")
    PaperlessBilling: str = _category("Yes or No")
    PaymentMethod: str = _category("Electronic check, Mailed check, Bank transfer or Credit card (both automatic)")
    MonthlyCharges: float = Field(ge=0, le=1000, description="Monthly charge")
    TotalCharges: float | None = Field(default=None, ge=0, description="Total charged so far; null for new customers")


class PredictResponse(BaseModel):
    """Calibrated churn probability and anything the caller should double-check."""

    churn_probability: float = Field(ge=0, le=1)
    model_version: str | None = Field(description="Registry version of the champion model that answered")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues, e.g. unseen categories")


def unknown_categories(features: CustomerFeatures) -> list[tuple[str, str]]:
    """(field, value) pairs whose category was never seen in training."""
    return [(f, getattr(features, f)) for f, known in KNOWN_CATEGORIES.items() if getattr(features, f) not in known]


class ApproveRequest(BaseModel):
    """Which challenger version the approver was looking at."""

    version: str = Field(min_length=1, max_length=20, pattern=r"^\d+$", description="Registry version number")
