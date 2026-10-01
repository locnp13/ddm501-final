"""Alert hub: Alertmanager webhook in, recent alerts out."""
import json

import pytest
from fastapi.testclient import TestClient

from src.alerts import hub


@pytest.fixture
def client():
    hub._active.clear()
    hub._history.clear()
    return TestClient(hub.app)


def alert(name="ApiDown", status="firing", severity="critical", fingerprint="f1", **extra) -> dict:
    return {
        "status": status,
        "labels": {"alertname": name, "severity": severity, "pod": "api-1"},
        "annotations": {"summary": f"{name} summary", "description": "details"},
        "startsAt": "2026-10-01T08:00:00Z",
        "endsAt": "0001-01-01T00:00:00Z" if status == "firing" else "2026-10-01T08:05:00Z",
        "generatorURL": "http://prometheus/graph",
        "fingerprint": fingerprint,
        **extra,
    }


def post(client, *alerts):
    # The real payload has more fields than the hub reads; keep a few to prove they are ignored.
    body = {"version": "4", "status": "firing", "receiver": "alert-hub", "alerts": list(alerts)}
    return client.post("/webhook", json=body)


def test_firing_alert_becomes_active(client) -> None:
    assert post(client, alert()).json() == {"received": 1}
    body = client.get("/alerts").json()
    assert body["counts"] == {"firing": 1, "critical": 1}
    first = body["active"][0]
    assert first["name"] == "ApiDown" and first["summary"] == "ApiDown summary"
    assert first["labels"] == {"pod": "api-1"}  # alertname and severity are lifted out of the labels


def test_resolved_alert_leaves_the_active_list_and_is_logged(client) -> None:
    post(client, alert())
    post(client, alert(status="resolved"))
    body = client.get("/alerts").json()
    assert body["active"] == [] and body["counts"]["firing"] == 0
    assert [e["status"] for e in body["history"]] == ["resolved", "firing"]  # newest first
    assert body["history"][0]["ended_at"] == "2026-10-01T08:05:00Z"


def test_repeated_notification_is_not_a_new_event(client) -> None:
    post(client, alert())
    post(client, alert())  # Alertmanager re-sends firing alerts every repeat_interval
    body = client.get("/alerts").json()
    assert body["counts"]["firing"] == 1 and len(body["history"]) == 1


def test_active_alerts_are_ordered_by_severity(client) -> None:
    post(client, alert("A", severity="info", fingerprint="a"), alert("B", severity="critical", fingerprint="b"),
         alert("C", severity="warning", fingerprint="c"))
    assert [a["name"] for a in client.get("/alerts").json()["active"]] == ["B", "C", "A"]


def test_history_is_capped(client) -> None:
    for i in range(hub.HISTORY_SIZE + 25):
        post(client, alert(fingerprint=f"f{i}"))
    assert len(hub._history) == hub.HISTORY_SIZE
    assert len(client.get("/alerts?limit=1000").json()["history"]) == hub.HISTORY_SIZE


@pytest.mark.parametrize(
    "bad",
    [
        {},  # no alerts key
        {"alerts": [{"status": "firing", "labels": {}}]},  # no fingerprint
        {"alerts": [{"status": "unknown", "fingerprint": "x"}]},
        {"alerts": [{"status": "firing", "fingerprint": "x"}] * 101},  # too many
    ],
)
def test_malformed_payload_is_rejected(client, bad) -> None:
    assert client.post("/webhook", json=bad).status_code == 422
    assert client.get("/alerts").json()["counts"]["firing"] == 0


def test_long_text_is_truncated(client) -> None:
    a = alert()
    a["annotations"]["summary"] = "x" * 5000
    post(client, a)
    assert len(client.get("/alerts").json()["active"][0]["summary"]) == 300


def test_health(client) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


class FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self, *_):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_firing_alerts_are_read_back_from_alertmanager_after_a_restart(client, monkeypatch) -> None:
    listed = [{
        "fingerprint": "abc",
        "labels": {"alertname": "ApiNoTargets", "severity": "critical"},
        "annotations": {"summary": "No pod"},
        "startsAt": "2026-10-01T09:30:03Z",
        "status": {"state": "active"},
    }]
    urls = []
    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda url, timeout: urls.append(url) or FakeResponse(listed))
    assert hub.sync_from_alertmanager("http://am:9093/alertmanager") == 1
    assert "silenced=false" in urls[0] and urls[0].startswith("http://am:9093/alertmanager/api/v2/alerts")
    body = client.get("/alerts").json()
    assert body["counts"] == {"firing": 1, "critical": 1} and body["active"][0]["summary"] == "No pod"
    assert body["history"] == []  # a sync is not a new event


def test_a_webhook_that_arrived_first_is_not_overwritten_by_the_sync(client, monkeypatch) -> None:
    post(client, alert(fingerprint="abc"))
    listed = [{"fingerprint": "abc", "labels": {"alertname": "Other"}, "annotations": {}, "startsAt": ""}]
    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda url, timeout: FakeResponse(listed))
    hub.sync_from_alertmanager("http://am")
    assert client.get("/alerts").json()["active"][0]["name"] == "ApiDown"
