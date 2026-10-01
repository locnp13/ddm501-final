"""Alert hub: receives Alertmanager webhooks and serves the recent alerts to the web UI.

Alertmanager POSTs to /webhook from inside the cluster (the Ingress exposes only GET /alerts). The
hub keeps the alerts that are firing now plus a short event log in memory. The event log is lost on a
restart, but the firing alerts are read back from Alertmanager at startup, so the list stays true.
"""
import json
import logging
import os
import threading
import time
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

HISTORY_SIZE = 200
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

logger = logging.getLogger("alert-hub")
# e.g. http://alertmanager:9093/alertmanager; empty disables the startup sync
ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL", "")
_lock = threading.Lock()
_active: dict[str, dict[str, Any]] = {}
_history: deque[dict[str, Any]] = deque(maxlen=HISTORY_SIZE)


def sync_from_alertmanager(url: str) -> int:
    """Add the alerts Alertmanager reports as active and not silenced; returns how many it listed."""
    query = "active=true&silenced=false&inhibited=false"
    with urllib.request.urlopen(f"{url}/api/v2/alerts?{query}", timeout=5) as response:  # noqa: S310 (fixed URL from config)
        items = json.load(response)
    with _lock:
        for item in items:
            alert = IncomingAlert(
                status="firing",
                labels=item.get("labels", {}),
                annotations=item.get("annotations", {}),
                startsAt=item.get("startsAt", ""),
                fingerprint=item["fingerprint"],
            )
            _active.setdefault(alert.fingerprint, _record(alert))  # a webhook that already arrived wins
    return len(items)


def _sync_when_ready(url: str) -> None:
    """Alertmanager may start after the hub, so retry for a while."""
    for _ in range(24):
        try:
            logger.info("Read %d active alerts from Alertmanager", sync_from_alertmanager(url))
            return
        except Exception as exc:
            logger.warning("Alertmanager not ready yet: %s", exc)
            time.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Fill the active list from Alertmanager in the background so the hub is ready at once."""
    if ALERTMANAGER_URL:
        threading.Thread(target=_sync_when_ready, args=(ALERTMANAGER_URL,), daemon=True).start()
    yield


app = FastAPI(title="Alert hub", version="0.1.0", lifespan=lifespan)


class IncomingAlert(BaseModel):
    """One alert of an Alertmanager webhook payload (version 4); unknown fields are ignored."""

    status: Literal["firing", "resolved"]
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    startsAt: str = ""
    endsAt: str | None = None
    fingerprint: str = Field(min_length=1, max_length=64)


class WebhookPayload(BaseModel):
    """Alertmanager groups alerts per notification; cap the size so a bad sender cannot flood the hub."""

    alerts: list[IncomingAlert] = Field(max_length=100)


def _record(alert: IncomingAlert) -> dict[str, Any]:
    text = alert.annotations
    return {
        "id": alert.fingerprint,
        "name": alert.labels.get("alertname", "unknown")[:100],
        "severity": alert.labels.get("severity", "warning")[:20],
        "status": alert.status,
        "summary": text.get("summary", "")[:300],
        "description": text.get("description", "")[:600],
        "started_at": alert.startsAt,
        "ended_at": alert.endsAt if alert.status == "resolved" else None,
        "labels": {k: v[:100] for k, v in alert.labels.items() if k not in ("alertname", "severity")},
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.post("/webhook")
def receive(payload: WebhookPayload) -> dict[str, int]:
    """Apply an Alertmanager notification: firing alerts become active, resolved ones leave the active list."""
    with _lock:
        for alert in payload.alerts:
            record = _record(alert)
            if alert.status == "firing":
                if alert.fingerprint not in _active:  # a repeated notification is not a new event
                    _history.append(record)
                _active[alert.fingerprint] = record
            else:
                _active.pop(alert.fingerprint, None)
                _history.append(record)
    return {"received": len(payload.alerts)}


@app.get("/alerts")
def alerts(limit: int = 50) -> dict[str, Any]:
    """Alerts firing now (most severe first) and the latest events, newest first."""
    with _lock:
        active = sorted(_active.values(), key=lambda a: (SEVERITY_ORDER.get(a["severity"], 9), a["started_at"]))
        history = list(_history)[::-1][: max(1, min(limit, HISTORY_SIZE))]
    return {
        "active": active,
        "history": history,
        "counts": {
            "firing": len(active),
            "critical": sum(1 for a in active if a["severity"] == "critical"),
        },
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness and readiness probe."""
    return {"status": "ok"}
