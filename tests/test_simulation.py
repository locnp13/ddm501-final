"""Tests for the traffic simulator used to demonstrate drift and validation alerts."""
import random

import pytest

from src.serving.schemas import CustomerFeatures
from src.simulation import traffic
from tests import test_api

champion_dir = test_api.champion_dir  # reuse the trained champion and client fixtures
client = test_api.client


@pytest.fixture(scope="module")
def customers():
    return traffic.load_customers()


def test_loaded_customers_match_the_api_schema(customers) -> None:
    assert len(customers) > 7000
    assert "Churn" not in customers[0]
    CustomerFeatures(**customers[0])  # raises if a field is missing or mistyped
    assert any(c["TotalCharges"] is None for c in customers)  # new customers keep a null total


def test_generation_is_reproducible(customers) -> None:
    assert traffic.generate(customers, "drift", 50, seed=3) == traffic.generate(customers, "drift", 50, seed=3)


def test_unknown_mode_is_refused(customers) -> None:
    with pytest.raises(ValueError):
        traffic.generate(customers, "chaos", 1)


def test_normal_traffic_stays_below_the_drift_threshold(customers) -> None:
    report = traffic.psi_report(customers, traffic.generate(customers, "normal", 2000))
    assert max(report.values()) < 0.1


def test_drift_traffic_crosses_the_alert_threshold_on_the_shifted_inputs(customers) -> None:
    report = traffic.psi_report(customers, traffic.generate(customers, "drift", 2000))
    assert {"tenure", "Contract", "MonthlyCharges", "PaymentMethod"} <= {f for f, v in report.items() if v > 0.2}
    assert report["gender"] < 0.05  # untouched inputs stay put


def test_drifted_customers_are_still_valid_requests(customers) -> None:
    rng = random.Random(0)
    for customer in customers[:200]:
        CustomerFeatures(**traffic.drift(customer, rng, strength=1.0))


def test_corrupted_customers_are_always_rejected_by_the_schema(customers) -> None:
    rng = random.Random(0)
    for customer in customers[:100]:
        with pytest.raises(ValueError):
            CustomerFeatures(**traffic.corrupt(customer, rng))


def test_invalid_mode_mixes_in_the_requested_share(customers) -> None:
    bodies = traffic.generate(customers, "invalid", 1000, invalid_share=0.3)
    bad = 0
    for body in bodies:
        try:
            CustomerFeatures(**body)
        except ValueError:
            bad += 1
    assert 0.25 < bad / len(bodies) < 0.35


def test_psi_is_zero_for_identical_samples_and_grows_with_shift() -> None:
    same = ["a"] * 50 + ["b"] * 50
    assert traffic.psi(same, same) == 0
    assert traffic.psi(same, ["a"] * 90 + ["b"] * 10) > traffic.psi(same, ["a"] * 60 + ["b"] * 40) > 0


def test_run_paces_requests_and_counts_statuses() -> None:
    sent, pauses = [], []

    def send(path, body, request_id):
        sent.append((path, request_id))
        return 200

    statuses = traffic.run([{"x": i} for i in range(5)], send, rate=10, sleep=pauses.append, log=lambda _: None)
    assert statuses == {200: 5}
    assert sent[0] == ("/v1/predict", "sim-000001")
    assert pauses == [0.1] * 5


def test_run_groups_bodies_for_the_batch_endpoint() -> None:
    calls = []
    traffic.run([{"x": i} for i in range(5)], lambda p, b, r: calls.append((p, len(b["customers"]))) or 200,
                rate=0, batch=2, log=lambda _: None)
    assert calls == [("/v1/predict/batch", 2), ("/v1/predict/batch", 2), ("/v1/predict/batch", 1)]


def test_unreachable_api_is_reported_once() -> None:
    messages = []
    statuses = traffic.run([{}] * 3, lambda *_: 0, rate=0, log=messages.append)
    assert statuses == {0: 3}
    assert len([m for m in messages if "not reachable" in m]) == 1


def test_http_sender_reports_unreachable_api_as_0() -> None:
    assert traffic.http_sender("http://127.0.0.1:9", timeout=0.5)("/v1/predict", {}, "sim-1") == 0


def test_report_mode_sends_nothing(capsys) -> None:
    assert traffic.main(["--mode", "drift", "--count", "300", "--report"], send=lambda *_: pytest.fail("sent")) == 0
    assert "<- drift" in capsys.readouterr().out


def test_simulated_traffic_reaches_the_api_and_its_monitoring(client) -> None:
    """End to end against the real app: drift traffic is scored and counted, invalid traffic gets 422."""

    def send(path, body, request_id):
        return client.post(path, json=body, headers={"X-Request-ID": request_id}).status_code

    assert traffic.main(["--mode", "drift", "--count", "20", "--rate", "0"], send=send) == 0
    metrics = client.get("/metrics").text
    assert 'churn_feature_values_total{bucket="Month-to-month",feature="Contract",model_version="1"}' in metrics

    statuses = traffic.run(
        traffic.generate(traffic.load_customers(), "invalid", 40, invalid_share=0.5), send, rate=0,
        log=lambda _: None,
    )
    assert set(statuses) == {200, 422}
