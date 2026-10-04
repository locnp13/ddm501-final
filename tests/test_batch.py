"""Tests for /v1/predict/batch: same model, same validation and same monitoring as /v1/predict."""
import pytest

from src.serving import app as app_module
from src.serving.schemas import MAX_BATCH
from tests import test_api
from tests.test_api import features

# Reuse the trained champion and API client of test_api (pytest finds fixtures by module attribute).
champion_dir = test_api.champion_dir
client = test_api.client


def test_batch_scores_every_customer_in_order(client) -> None:
    customers = [features(tenure=1, Contract="Month-to-month"), features(tenure=70, Contract="Two year")]
    res = client.post("/v1/predict/batch", json={"customers": customers})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] == 2 and body["model_version"] == "1"
    probs = [p["churn_probability"] for p in body["predictions"]]
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_batch_matches_single_predictions(client) -> None:
    customers = [features(tenure=t) for t in (0, 5, 30)]
    batch = client.post("/v1/predict/batch", json={"customers": customers}).json()["predictions"]
    single = [client.post("/v1/predict", json=c).json()["churn_probability"] for c in customers]
    assert [p["churn_probability"] for p in batch] == pytest.approx(single)


def test_unseen_category_warns_only_for_that_customer(client) -> None:
    customers = [features(), {**features(), "PaymentMethod": "Momo"}]
    preds = client.post("/v1/predict/batch", json={"customers": customers}).json()["predictions"]
    assert preds[0]["warnings"] == []
    assert preds[1]["warnings"] == ["PaymentMethod: unseen category 'Momo'"]


def test_one_invalid_customer_rejects_the_batch_and_names_its_position(client) -> None:
    before = client.get("/metrics").text.count("churn_predictions_total{")
    res = client.post("/v1/predict/batch", json={"customers": [features(), {**features(), "tenure": -1}]})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"][-3:] == ["customers", 1, "tenure"]
    assert client.get("/metrics").text.count("churn_predictions_total{") == before
    assert 'endpoint="predict/batch",model_version="1",status="422"' in client.get("/metrics").text


@pytest.mark.parametrize("size", [0, MAX_BATCH + 1])
def test_batch_size_is_limited(client, size) -> None:
    res = client.post("/v1/predict/batch", json={"customers": [features()] * size})
    assert res.status_code == 422


def test_batch_is_counted_in_metrics(client) -> None:
    client.post("/v1/predict/batch", json={"customers": [features()] * 3})
    text = client.get("/metrics").text
    assert 'churn_requests_total{endpoint="predict/batch",model_version="1",status="200"}' in text
    assert "churn_batch_size_count" in text


def test_batch_without_model_gives_503(client, monkeypatch) -> None:
    monkeypatch.setitem(app_module.state, "model", None)
    assert client.post("/v1/predict/batch", json={"customers": [features()]}).status_code == 503


def test_batch_failure_gives_500(client, monkeypatch) -> None:
    class Broken:
        def predict(self, frame):
            raise RuntimeError("boom")

    monkeypatch.setitem(app_module.state, "model", Broken())
    assert client.post("/v1/predict/batch", json={"customers": [features()]}).status_code == 500


def test_openapi_documents_the_batch_endpoint(client) -> None:
    spec = client.get("/openapi.json").json()
    assert "/v1/predict/batch" in spec["paths"]
    batch = spec["components"]["schemas"]["BatchRequest"]
    assert batch["properties"]["customers"]["maxItems"] == MAX_BATCH
    assert batch["examples"][0]["customers"]
