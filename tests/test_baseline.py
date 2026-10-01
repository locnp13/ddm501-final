"""Feature baseline: bucketing and the shares saved with the model."""
import math

import pandas as pd
import pytest

from src.training.baseline import build_baseline
from src.training.buckets import MISSING, OTHER, bucket_label, numeric_labels
from src.training.data import clean, coerce_total_charges, validate
from tests.conftest import synthetic_telco


@pytest.fixture
def X() -> pd.DataFrame:
    return clean(validate(coerce_total_charges(synthetic_telco(n=600)))).drop(columns=["Churn"])


def test_labels_are_sortable_and_cover_every_interval() -> None:
    assert numeric_labels([12, 24, 48]) == ["00_<12", "01_12-24", "02_24-48", "03_>=48"]


@pytest.mark.parametrize(
    ("value", "label"),
    [(0, "00_<12"), (11.99, "00_<12"), (12, "01_12-24"), (47, "02_24-48"), (48, "03_>=48"), (1e9, "03_>=48"),
     (None, MISSING), (math.nan, MISSING), ("abc", MISSING), ("30", "02_24-48")],
)
def test_numeric_bucket_edges(value, label) -> None:
    assert bucket_label({"type": "numeric", "cuts": [12, 24, 48]}, value) == label


def test_unseen_category_goes_to_other() -> None:
    spec = {"type": "categorical", "shares": {"Yes": 0.4, "No": 0.6}}
    assert bucket_label(spec, "Yes") == "Yes" and bucket_label(spec, "Momo") == OTHER


def test_every_input_column_has_shares_that_sum_to_one(X) -> None:
    baseline = build_baseline(X)
    assert baseline["rows"] == 600 and set(baseline["features"]) == set(X.columns)
    for name, spec in baseline["features"].items():
        assert sum(spec["shares"].values()) == pytest.approx(1.0, abs=1e-3), name


def test_numeric_features_get_quantile_buckets_and_a_missing_share(X) -> None:
    tenure = build_baseline(X)["features"]["tenure"]
    assert tenure["type"] == "numeric" and 3 <= len(tenure["cuts"]) <= 9
    assert all(0.04 < tenure["shares"][label] < 0.2 for label in numeric_labels(tenure["cuts"]))  # about deciles
    total = build_baseline(X)["features"]["TotalCharges"]
    assert total["shares"][MISSING] > 0  # new customers have no TotalCharges


def test_a_training_row_lands_in_a_bucket_that_exists_in_the_baseline(X) -> None:
    baseline = build_baseline(X)["features"]
    for _, row in X.head(50).iterrows():
        for name, spec in baseline.items():
            label = bucket_label(spec, row[name])
            assert label in spec["shares"] or label == OTHER, (name, label)


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("baseline-test")  # earlier tests leave another store's experiment id active
    return tmp_path


def make_run(data_md5: str) -> str:
    import mlflow

    with mlflow.start_run() as run:
        mlflow.set_tag("data_md5", data_md5)
    return run.info.run_id


def test_backfill_refuses_a_run_trained_on_other_data(mlflow_store) -> None:
    from src.training import baseline

    run_id = make_run("md5-of-some-older-data")  # the current data hash is "unknown": there is no dvc.lock here
    with pytest.raises(SystemExit, match="not backfilling"):
        baseline.backfill(run_id)


def test_backfill_logs_the_training_split_baseline(mlflow_store, monkeypatch) -> None:
    import mlflow.artifacts

    from src.training import baseline

    csv = mlflow_store / "churn.csv"
    clean(validate(coerce_total_charges(synthetic_telco(n=400)))).to_csv(csv, index=False)
    monkeypatch.setattr(baseline, "PROCESSED_PATH", csv)
    monkeypatch.setattr(baseline, "load_params", lambda: {"train": {"seed": 42, "test_size": 0.25}})
    run_id = make_run("unknown")
    baseline.backfill(run_id)
    logged = mlflow.artifacts.load_dict(f"runs:/{run_id}/{baseline.BASELINE_FILE}")
    assert logged["rows"] == 300 and "Contract" in logged["features"]  # 75% of 400 rows


def test_the_serving_app_imports_without_training_only_libraries() -> None:
    """The API image has no pandera, optuna, evidently or dvc; importing the app must not need them."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "for name in ('pandera', 'optuna', 'evidently', 'dvc'):\n"
        "    sys.modules[name] = None  # makes `import name` raise ImportError\n"
        "import src.serving.app\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "ok", result.stderr[-600:]
