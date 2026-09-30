"""Feature engineering as a sklearn transformer, so the API applies the same steps as training."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
SENSITIVE_COLS = ["gender", "SeniorCitizen"]
AUTOPAY_METHODS = {"Bank transfer (automatic)", "Credit card (automatic)"}
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerce TotalCharges, optionally add derived features and drop sensitive columns.

    The input keeps the raw Telco columns (the API contract); derived columns and the
    sensitive-column drop happen inside the pipeline.
    """

    def __init__(self, add_features: bool = True, drop_sensitive: bool = False) -> None:
        self.add_features = add_features
        self.drop_sensitive = drop_sensitive

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if self.add_features:
            out["tenure_bucket"] = pd.cut(out["tenure"], TENURE_BINS, labels=TENURE_LABELS).astype(object)
            out["num_services"] = (out[SERVICE_COLS] == "Yes").sum(axis=1)
            out["avg_charge_per_month"] = out["TotalCharges"] / out["tenure"].replace(0, np.nan)
            out["autopay"] = out["PaymentMethod"].isin(AUTOPAY_METHODS).astype(int)
            out["has_internet_no_support"] = (
                (out["InternetService"] != "No") & (out["OnlineSecurity"] != "Yes") & (out["TechSupport"] != "Yes")
            ).astype(int)
        if self.drop_sensitive:
            out = out.drop(columns=SENSITIVE_COLS)
        return out
