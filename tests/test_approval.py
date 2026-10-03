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
    app_module.state.update({"model": None, "info": None, "figures": {}, "profile": None})
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


def test_wrong_or_missing_key_is_rejected(client, pending, caplog) -> None:
    for headers in ({}, {"X-Admin-Key": "nope"}, {"X-Admin-Key": b"\xff\xfe"}):
        assert client.post("/v1/model/promote", json={"version": "2"}, headers=headers).status_code == 401
    assert alias_version(pending, CHAMPION) == "1"
    denied = [m for m in caplog.messages if "admin action=approve denied: invalid admin key" in m]
    assert len(denied) == 3 and not any("nope" in m for m in caplog.messages)
    assert 'churn_admin_actions_total{action="approve",result="denied"}' in client.get("/metrics").text


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
    assert 'churn_model_changes_total{kind="approve"} 1.0' in client.get("/metrics").text
    assert 'churn_admin_actions_total{action="approve",result="ok"} 1.0' in client.get("/metrics").text
    assert client.get("/v1/model/challenger").status_code == 404
    again = client.post("/v1/model/promote", json={"version": "2"}, headers={"X-Admin-Key": KEY})
    assert again.status_code == 404  # nothing left to approve


def test_version_history_lists_every_version(client) -> None:
    versions = client.get("/v1/model/versions").json()
    assert [v["version"] for v in versions] == ["2", "1"]  # newest first, nothing deleted
    assert versions[0]["aliases"] == [CHAMPION] and versions[0]["approved_at"]
    assert versions[1]["aliases"] == [] and versions[1]["pr_auc"] is not None


def test_rollback_needs_the_key_and_a_real_other_version(client, pending) -> None:
    body = {"version": "1"}
    assert client.post("/v1/model/rollback", json=body).status_code == 401
    assert client.post("/v1/model/rollback", json={"version": "99"}, headers={"X-Admin-Key": KEY}).status_code == 404
    assert client.post("/v1/model/rollback", json={"version": "2"}, headers={"X-Admin-Key": KEY}).status_code == 409
    assert alias_version(pending, CHAMPION) == "2"
    assert 'churn_admin_actions_total{action="rollback",result="failed"} 2.0' in client.get("/metrics").text


def test_incompatible_version_is_refused_and_nothing_changes(client, pending, monkeypatch) -> None:
    class Broken:
        def predict(self, _):
            raise ValueError("columns do not match")

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: Broken())
    res = client.post("/v1/model/rollback", json={"version": "1"}, headers={"X-Admin-Key": KEY})
    assert res.status_code == 422 and "nothing changed" in res.json()["detail"]
    assert alias_version(pending, CHAMPION) == "2"


def test_rollback_serves_the_older_version_at_once(client, pending) -> None:
    res = client.post("/v1/model/rollback", json={"version": "1"}, headers={"X-Admin-Key": KEY})
    assert res.status_code == 200, res.text
    assert alias_version(pending, CHAMPION) == "1"
    assert "restored_at" in pending.get_model_version(MODEL_NAME, "1").tags
    assert client.post("/v1/predict", json=features()).json()["model_version"] == "1"
    metrics = client.get("/metrics").text
    assert 'churn_model_changes_total{kind="rollback"} 1.0' in metrics
    assert 'churn_model_info{version="1"} 1.0' in metrics and 'churn_model_info{version="2"}' not in metrics
    assert [v["aliases"] for v in client.get("/v1/model/versions").json()] == [[], [CHAMPION]]


def test_replicas_follow_the_registry_alias(client, pending) -> None:
    assert client.get("/v1/model").json()["version"] == "1"
    assert app_module.sync_champion() is False  # nothing moved
    pending.set_registered_model_alias(MODEL_NAME, CHAMPION, "2")  # changed elsewhere, e.g. the MLflow UI
    assert app_module.sync_champion() is True
    assert client.get("/v1/model").json()["version"] == "2"
    assert client.post("/v1/predict", json=features()).json()["model_version"] == "2"
    assert app_module.sync_champion() is False


def test_pinned_instance_serves_one_version_and_cannot_change_it(pending, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "MODEL_URI", f"models:/{MODEL_NAME}/1")
    monkeypatch.setattr(app_module, "ADMIN_KEY", KEY)
    app_module.state.update({"model": None, "info": None, "figures": {}, "profile": None})
    with TestClient(app_module.app) as c:
        info = c.get("/v1/model").json()
        assert info["version"] == "1" and info["alias"] is None
        assert c.post("/v1/predict", json=features()).json()["model_version"] == "1"
        for path in ("/v1/model/promote", "/v1/model/rollback"):
            res = c.post(path, json={"version": "2"}, headers={"X-Admin-Key": KEY})
            assert res.status_code == 409 and "pinned" in res.json()["detail"]
        assert app_module.sync_champion() is False
    assert alias_version(pending, CHAMPION) == "2"
