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


def test_ready_requires_a_loaded_model(client) -> None:
    assert client.get("/ready").json() == {"status": "ready", "model_version": "1"}


def test_predict_returns_probability(client) -> None:
    res = client.post("/v1/predict", json=features())
    assert res.status_code == 200, res.text
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


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"Contract": None}, "Contract"),  # null where a value is required
        ({"tenure": -1}, "tenure"),
        ({"tenure": 5.5}, "tenure"),  # not a whole number of months
        ({"tenure": "five"}, "tenure"),
        ({"MonthlyCharges": -0.01}, "MonthlyCharges"),
        ({"TotalCharges": -5}, "TotalCharges"),
        ({"SeniorCitizen": 2}, "SeniorCitizen"),
        ({"gender": ""}, "gender"),
    ],
)
def test_invalid_values_are_rejected_with_422(client, change, field) -> None:
    res = client.post("/v1/predict", json={**features(), **change})
    assert res.status_code == 422
    assert field in {err["loc"][-1] for err in res.json()["detail"]}


def test_missing_field_is_rejected(client) -> None:
    body = features()
    del body["PaymentMethod"]
    res = client.post("/v1/predict", json=body)
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"][-1] == "PaymentMethod"


def test_non_json_body_is_rejected(client) -> None:
    res = client.post("/v1/predict", content=b"not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_rejected_requests_never_reach_the_model(client) -> None:
    before = client.get("/metrics").text.count("churn_predictions_total{")
    client.post("/v1/predict", json={**features(), "tenure": -1})
    assert client.get("/metrics").text.count("churn_predictions_total{") == before


def test_422_is_counted_in_metrics(client) -> None:
    client.post("/v1/predict", json={**features(), "tenure": -1})
    assert 'churn_requests_total{endpoint="predict",status="422"}' in client.get("/metrics").text


def test_unseen_category_is_scored_with_warning(client) -> None:
    res = client.post("/v1/predict", json=features(PaymentMethod="Momo"))
    assert res.status_code == 200
    assert res.json()["warnings"] == ["PaymentMethod: unseen category 'Momo'"]
    assert 'churn_unknown_category_total{field="PaymentMethod"}' in client.get("/metrics").text


def test_known_categories_give_no_warnings_and_report_version(client) -> None:
    body = client.post("/v1/predict", json=features()).json()
    assert body["warnings"] == [] and body["model_version"] == "1"


def test_extra_fields_are_ignored(client) -> None:
    assert client.post("/v1/predict", json={**features(), "customerID": "0001-A"}).status_code == 200


def test_openapi_documents_example_and_constraints(client) -> None:
    schema = client.get("/openapi.json").json()["components"]["schemas"]["CustomerFeatures"]
    assert schema["examples"][0]["Contract"] == "Month-to-month"
    assert schema["properties"]["tenure"]["minimum"] == 0
    assert set(schema["required"]) == set(schema["properties"]) - {"TotalCharges"}


def test_api_schema_matches_training_schema() -> None:
    """The API must accept exactly the columns the model was trained on, with the same categories."""
    from src.serving.schemas import KNOWN_CATEGORIES, CustomerFeatures
    from src.training.data import SCHEMA

    assert set(CustomerFeatures.model_fields) == set(SCHEMA.columns) - {"customerID", "Churn"}
    categorical = {
        name for name, col in SCHEMA.columns.items()
        if col.checks and "allowed_values" in col.checks[0].statistics
    } - {"Churn", "SeniorCitizen"}
    assert categorical == set(KNOWN_CATEGORIES)
    for name, known in KNOWN_CATEGORIES.items():
        assert set(SCHEMA.columns[name].checks[0].statistics["allowed_values"]) == set(known), name


def test_no_model_gives_503(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/empty.db")
    monkeypatch.setattr(app_module, "MODEL_URI", "models:/churn-model@champion")
    app_module.state.update({"model": None, "info": None, "figures": {}})
    with TestClient(app_module.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/ready").status_code == 503  # alive, but not ready for traffic
        assert c.post("/v1/predict", json=features()).status_code == 503
        assert c.get("/v1/model").status_code == 503
