"""Feature baseline: how each input was distributed in the training data, for drift monitoring.

The baseline is saved with the model as `feature_baseline.json`. At serving time every request is
counted into the same buckets (see src/serving/drift.py) and Prometheus compares the live shares
with these training shares (PSI). Only bucket counts are kept, never customer rows.

Numeric inputs get quantile buckets, categorical inputs one bucket per value seen in training, and
anything unseen at serving time falls into `other`.
"""
import sys
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.training.buckets import BASELINE_FILE, MISSING, NUMERIC, bucket_label, numeric_labels
from src.training.data import PROCESSED_PATH, load_params
from src.training.registry import CHAMPION, MODEL_NAME, provenance


def build_baseline(X: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    """Per-feature shares of the training data (rows of raw input columns, without the target)."""
    features: dict[str, Any] = {}
    for column in X.columns:
        if column in NUMERIC:
            values = pd.to_numeric(X[column], errors="coerce")
            observed = values.dropna().to_numpy()
            quantiles = np.quantile(observed, np.linspace(0, 1, bins + 1))
            cuts = sorted({round(float(q), 2) for q in quantiles[1:-1]})
            spec: dict[str, Any] = {"type": "numeric", "cuts": cuts}
            labels = [bucket_label({**spec, "shares": {}}, v) for v in observed]
            shares = pd.Series(labels).value_counts().div(len(values)).to_dict()
            shares.update({label: shares.get(label, 0.0) for label in numeric_labels(cuts)})
            shares[MISSING] = float(values.isna().mean())
        else:
            shares = X[column].astype(str).value_counts(normalize=True).to_dict()
            spec = {"type": "categorical"}
        features[column] = {**spec, "shares": {k: round(float(v), 6) for k, v in shares.items()}}
    return {"rows": int(len(X)), "features": features}


def backfill(run_id: str | None = None) -> None:
    """Log the baseline of the training split into an existing run (default: the champion's run).

    For models trained before the baseline was logged. Refuses if the data changed since the run,
    because the baseline would then describe other data than the model saw.
    """
    client = MlflowClient()
    run_id = run_id or client.get_model_version_by_alias(MODEL_NAME, CHAMPION).run_id
    run = client.get_run(run_id)
    expected, current = run.data.tags.get("data_md5"), provenance()["data_md5"]
    if expected != current:
        sys.exit(f"Run {run_id} was trained on data {expected}, but the current data is {current}; not backfilling.")
    train_p = load_params()["train"]
    df = pd.read_csv(PROCESSED_PATH)
    from sklearn.model_selection import train_test_split  # same split as the training stage

    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_tr, _, _, _ = train_test_split(X, y, test_size=train_p["test_size"], stratify=y, random_state=train_p["seed"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(build_baseline(X_tr), BASELINE_FILE)
        mlflow.set_tag("feature_baseline", "backfilled")
    print(f"Logged {BASELINE_FILE} ({len(X_tr)} training rows) to run {run_id}")


if __name__ == "__main__":
    backfill(sys.argv[1] if len(sys.argv) > 1 else None)
