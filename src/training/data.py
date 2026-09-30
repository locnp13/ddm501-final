"""Ingest the Telco churn dataset: download, validate the schema, clean, write."""
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pandera as pa
import yaml

RAW_PATH = Path("data/raw/telco.csv")
PROCESSED_PATH = Path("data/processed/churn.csv")
PARAMS_PATH = Path("params.yaml")

_YES_NO = ["Yes", "No"]
_INTERNET_ADDON = ["Yes", "No", "No internet service"]

SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, unique=True),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        "Partner": pa.Column(str, pa.Check.isin(_YES_NO)),
        "Dependents": pa.Column(str, pa.Check.isin(_YES_NO)),
        "tenure": pa.Column(int, pa.Check.ge(0), coerce=True),
        "PhoneService": pa.Column(str, pa.Check.isin(_YES_NO)),
        "MultipleLines": pa.Column(str, pa.Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(_INTERNET_ADDON)),
        "OnlineBackup": pa.Column(str, pa.Check.isin(_INTERNET_ADDON)),
        "DeviceProtection": pa.Column(str, pa.Check.isin(_INTERNET_ADDON)),
        "TechSupport": pa.Column(str, pa.Check.isin(_INTERNET_ADDON)),
        "StreamingTV": pa.Column(str, pa.Check.isin(_INTERNET_ADDON)),
        "StreamingMovies": pa.Column(str, pa.Check.isin(_INTERNET_ADDON)),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(_YES_NO)),
        "PaymentMethod": pa.Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0), coerce=True),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "Churn": pa.Column(str, pa.Check.isin(_YES_NO)),
    },
    strict=True,
)


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Read pipeline parameters from params.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def download(url: str, dest: Path = RAW_PATH) -> None:
    """Download the CSV to `dest`; keep an existing copy if the download fails."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        if not dest.exists():
            raise SystemExit(f"Cannot download {url} and no cached {dest}: {exc}") from exc
        print(f"Download failed ({exc}); using cached {dest}", file=sys.stderr)


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Turn blank TotalCharges strings (new customers) into NaN."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate against SCHEMA; raises pandera.errors.SchemaError(s) on violation."""
    return SCHEMA.validate(df, lazy=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the identifier and encode the target as 0/1."""
    out = df.drop(columns=["customerID"])
    out["Churn"] = (out["Churn"] == "Yes").astype(int)
    return out


def main() -> None:
    """DVC `ingest` stage."""
    params = load_params()
    download(params["data"]["url"])
    df = pd.read_csv(RAW_PATH, dtype={"TotalCharges": "object"})
    if "TotalCharges" in df.columns:
        df = coerce_total_charges(df)
    try:
        df = validate(df)
    except pa.errors.SchemaErrors as exc:
        print(f"Schema validation failed:\n{exc.failure_cases}", file=sys.stderr)
        raise SystemExit(1) from exc
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean(df).to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {PROCESSED_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
