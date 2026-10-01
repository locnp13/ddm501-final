"""API tests against a real (tiny) trained model in a temporary MLflow store."""
import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from src.serving import app as app_module
from src.training.data import clean, coerce_total_charges, validate
from src.training.registry import promote
from src.training.train import run_training
from tests.conftest import synthetic_telco
from tests.test_data import make_row
from tests.test_train_smoke import PARAMS


@pytest.fixture(scope="module")
def champion_dir(tmp_path_factory):
    """Train once, promote to champion, and point MLflow at that store for the module."""
    tmp = tmp_path_factory.mktemp("mlflow")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
        df = clean(validate(coerce_total_charges(synthetic_telco())))
        run_training(df, PARAMS)
        promote(MlflowClient())
        yield tmp


@pytest.fixture
def client(champion_dir, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_URI", "models:/churn-model@champion")
    app_module.state.update({"model": None, "info": None, "figures": {}})
    with TestClient(app_module.app) as c:
        yield c


def features(**overrides) -> dict:
    row = make_row(**overrides)
    row.pop("customerID")
    row.pop("Churn")
    row["TotalCharges"] = float(row["TotalCharges"])
    return row


def test_health_reports_loaded_model(client) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability(client) -> None:
    res = client.post("/v1/predict", json=features())
    assert res.status_code == 200
    assert 0.0 <= res.json()["churn_probability"] <= 1.0


def test_predict_accepts_null_total_charges(client) -> None:
    res = client.post("/v1/predict", json={**features(tenure=0), "TotalCharges": None})
    assert res.status_code == 200


def test_model_info_describes_champion(client) -> None:
    info = client.get("/v1/model").json()
    assert info["name"] == "churn-model" and info["alias"] == "champion"
    assert {"pr_auc", "brier", "best_profit"} <= info["metrics"].keys()
    assert info["selected"]["model"] in {"dummy", "logreg", "xgb"}
    assert info["economics"]["retention_success"] == 0.3
    assert len(info["profit_curve"]) == 3  # k_grid of the test params
    assert info["candidates"][0]["cv_pr_auc"] >= info["candidates"][-1]["cv_pr_auc"]


def test_model_figures(client) -> None:
    png = client.get("/v1/model/figures/calibration.png")
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
    assert client.get("/v1/model/figures/../../etc/passwd").status_code == 404
    assert client.get("/v1/model/figures/other.png").status_code == 404


def test_metrics_endpoint_counts_predictions(client) -> None:
    client.post("/v1/predict", json=features())
    assert "churn_requests_total" in client.get("/metrics").text


def test_no_model_gives_503(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/empty.db")
    monkeypatch.setattr(app_module, "MODEL_URI", "models:/churn-model@champion")
    app_module.state.update({"model": None, "info": None, "figures": {}})
    with TestClient(app_module.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/v1/predict", json=features()).status_code == 503
        assert c.get("/v1/model").status_code == 503
