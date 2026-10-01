"""Approving a challenger through the API. Tests share one store and run in order: the last one promotes."""
import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from src.serving import app as app_module
from src.training.data import clean, coerce_total_charges, validate
from src.training.registry import CHALLENGER, CHAMPION, MODEL_NAME, promote
from src.training.train import run_training
from tests.conftest import synthetic_telco
from tests.test_api import features
from tests.test_train_smoke import PARAMS

KEY = "test-admin-key"


@pytest.fixture(scope="module")
def pending(tmp_path_factory):
    """Registry with champion v1 and a challenger v2 waiting for approval."""
    tmp = tmp_path_factory.mktemp("approval")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
        df = clean(validate(coerce_total_charges(synthetic_telco())))
        run_training(df, PARAMS)
        promote(MlflowClient())
        assert run_training(df, PARAMS)["version"] == "2"
        yield MlflowClient()


@pytest.fixture
def client(pending, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_URI", f"models:/{MODEL_NAME}@{CHAMPION}")
    monkeypatch.setattr(app_module, "ADMIN_KEY", KEY)
    app_module.state.update({"model": None, "info": None, "figures": {}})
    with TestClient(app_module.app) as c:
        yield c


def alias_version(pending, alias: str) -> str | None:
    try:
        return str(pending.get_model_version_by_alias(MODEL_NAME, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def test_challenger_is_described_next_to_the_champion(client) -> None:
    challenger = client.get("/v1/model/challenger").json()
    assert challenger["alias"] == CHALLENGER and challenger["version"] == "2"
    assert client.get("/v1/model").json()["version"] == "1"


def test_wrong_or_missing_key_is_rejected(client, pending) -> None:
    for headers in ({}, {"X-Admin-Key": "nope"}, {"X-Admin-Key": b"\xff\xfe"}):
        assert client.post("/v1/model/promote", json={"version": "2"}, headers=headers).status_code == 401
    assert alias_version(pending, CHAMPION) == "1"


def test_approval_is_disabled_without_a_configured_key(client, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "ADMIN_KEY", "")
    res = client.post("/v1/model/promote", json={"version": "2"}, headers={"X-Admin-Key": ""})
    assert res.status_code == 503


@pytest.mark.parametrize("body", [{}, {"version": ""}, {"version": "v2"}, {"version": "2; drop"}])
def test_malformed_version_is_rejected(client, body) -> None:
    assert client.post("/v1/model/promote", json=body, headers={"X-Admin-Key": KEY}).status_code == 422


def test_stale_version_is_refused_and_nothing_changes(client, pending) -> None:
    res = client.post("/v1/model/promote", json={"version": "1"}, headers={"X-Admin-Key": KEY})
    assert res.status_code == 409
    assert alias_version(pending, CHAMPION) == "1" and alias_version(pending, CHALLENGER) == "2"
    assert client.get("/v1/model").json()["version"] == "1"


def test_approval_serves_the_new_model_without_restart(client, pending) -> None:
    assert client.post("/v1/predict", json=features()).json()["model_version"] == "1"
    res = client.post("/v1/model/promote", json={"version": "2"}, headers={"X-Admin-Key": KEY})
    assert res.status_code == 200, res.text
    assert res.json()["version"] == "2" and res.json()["alias"] == CHAMPION

    assert alias_version(pending, CHAMPION) == "2" and alias_version(pending, CHALLENGER) is None
    assert "approved_at" in pending.get_model_version(MODEL_NAME, "2").tags
    assert client.get("/v1/model").json()["version"] == "2"
    assert client.post("/v1/predict", json=features()).json()["model_version"] == "2"
    assert "churn_model_promotions_total 1.0" in client.get("/metrics").text
    assert client.get("/v1/model/challenger").status_code == 404
    again = client.post("/v1/model/promote", json={"version": "2"}, headers={"X-Admin-Key": KEY})
    assert again.status_code == 404  # nothing left to approve
