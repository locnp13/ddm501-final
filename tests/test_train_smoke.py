"""Smoke test: run the whole training flow on small synthetic data with a temporary MLflow store."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.training.registry import CHALLENGER, CHAMPION, MODEL_NAME, promote
from src.training.train import run_training, select_config

PARAMS = {
    "train": {
        "seed": 42, "test_size": 0.25, "n_trials": 2, "cv_folds": 3,
        "models": ["dummy", "logreg", "xgb"], "sensitive_tolerance": 0.005,
    },
    "gate": {"min_pr_auc": 0.0, "baseline_margin": -1.0, "champion_tolerance": 1.0},
    "economics": {"value_months": 12, "retention_cost_rate": 0.1, "retention_success": 0.3, "k_grid": [0.1, 0.3, 0.5]},
}


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield MlflowClient()


def test_training_registers_challenger_and_promotes(telco_clean, mlflow_sqlite) -> None:
    outcome = run_training(telco_clean, PARAMS)
    assert outcome["passed"], outcome["reasons"]
    assert 0.0 <= outcome["brier"] <= 1.0
    assert outcome["pr_auc"] > 0
    assert 0.0 < outcome["test_churn_rate"] < 1.0
    assert outcome["selected"]["model"] in PARAMS["train"]["models"]
    assert isinstance(outcome["selected"]["add_features"], bool)
    assert isinstance(outcome["selected"]["drop_sensitive"], bool)
    assert str(mlflow_sqlite.get_model_version_by_alias(MODEL_NAME, CHALLENGER).version) == outcome["version"]

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{outcome['version']}")
    proba = model.predict(telco_clean.drop(columns=["Churn"]).head(5))
    assert ((proba >= 0) & (proba <= 1)).all()

    promote(mlflow_sqlite)
    assert str(mlflow_sqlite.get_model_version_by_alias(MODEL_NAME, CHAMPION).version) == outcome["version"]


def test_gate_failure_registers_nothing(telco_clean, mlflow_sqlite) -> None:
    params = {**PARAMS, "gate": {"min_pr_auc": 0.999, "baseline_margin": 0.0, "champion_tolerance": 0.0}}
    outcome = run_training(telco_clean, params)
    assert not outcome["passed"] and outcome["version"] is None
    assert mlflow_sqlite.search_registered_models() == []


def test_select_config_prefers_dropping_sensitive_within_tolerance() -> None:
    def row(drop: bool, score: float) -> dict:
        return {"model": "xgb", "add_features": True, "drop_sensitive": drop, "params": {}, "cv_pr_auc": score}

    assert select_config([row(False, 0.660), row(True, 0.657)], 0.005)["drop_sensitive"] is True
    assert select_config([row(False, 0.660), row(True, 0.640)], 0.005)["drop_sensitive"] is False
