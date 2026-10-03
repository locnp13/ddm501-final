"""Churn prediction API: health, Prometheus metrics, and model-backed prediction."""
import hmac
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.serving.drift import FEATURE_BASELINE, load_profile
from src.serving.model_info import FIGURES, list_versions, load_figure, load_model_info, parse_model_uri
from src.serving.schemas import EXAMPLE, CustomerFeatures, PredictResponse, VersionRequest, unknown_categories
from src.training.registry import CHALLENGER, CHAMPION, MODEL_NAME

logger = logging.getLogger("churn-api")

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
QUIET_PATHS = {"/health", "/ready", "/metrics"}  # probes and scrapes hit these every few seconds: DEBUG only


class RequestIdFilter(logging.Filter):
    """Stamp every record with the id of the request being served, so one call can be followed across log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    """Log to stderr with time, level, logger and request id; LOG_LEVEL (default INFO) sets the threshold.

    uvicorn only configures its own loggers, so without this the app's INFO lines were dropped
    (root level WARNING) and warnings had no timestamp.
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
    )
    for handler in logging.getLogger().handlers:  # the format needs the field on every record these handlers see
        handler.addFilter(RequestIdFilter())
    logger.addFilter(RequestIdFilter())  # and the app's own records carry it whatever handler reads them


configure_logging()

MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")  # empty disables model approval through the API
# > 0: follow the registry alias, so several replicas converge after an approval or rollback.
MODEL_POLL_SECONDS = float(os.getenv("MODEL_POLL_SECONDS", "0"))

REQUESTS = Counter(
    "churn_requests_total", "Requests by endpoint, status and model version", ["endpoint", "status", "model_version"]
)
LATENCY = Histogram("churn_request_latency_seconds", "Request latency", ["endpoint", "model_version"])
PREDICTIONS = Counter("churn_predictions_total", "Predictions by class and model version", ["label", "model_version"])
CHURN_PROBABILITY = Histogram(
    "churn_probability", "Predicted churn probability", ["model_version"], buckets=[i / 10 for i in range(11)]
)
MODEL_LOADED = Gauge("churn_model_loaded", "1 if a model is loaded, else 0")
MODEL_INFO = Gauge("churn_model_info", "1 for the model version this instance serves", ["version"])
MODEL_CHANGES = Counter("churn_model_changes_total", "Model switches by kind", ["kind"])
ADMIN_ACTIONS = Counter(
    "churn_admin_actions_total",
    "Model approval and rollback attempts by outcome (ok, denied, failed)",
    ["action", "result"],
)
UNKNOWN_CATEGORY = Counter(
    "churn_unknown_category_total", "Requests with a category never seen in training", ["field"]
)

state: dict[str, Any] = {"model": None, "info": None, "figures": {}, "profile": None}
promote_lock = threading.Lock()


def is_pinned() -> bool:
    """True when MODEL_URI names one version (models:/churn-model/3) instead of an alias.

    A pinned instance never follows or changes the registry; this is how canary and A/B
    deployments run two versions side by side.
    """
    parsed = parse_model_uri(MODEL_URI)
    return parsed is not None and parsed[1] is None


def set_served(model: Any, info: dict[str, Any] | None) -> None:
    """Make `model` the one answering requests and refresh the gauges that describe it."""
    profile = load_profile(info)
    state.update(model=model, info=info, figures={}, profile=profile)
    MODEL_LOADED.set(1 if model is not None else 0)
    MODEL_INFO.clear()
    FEATURE_BASELINE.clear()
    if info is not None:
        MODEL_INFO.labels(info["version"]).set(1)
    if profile is not None:
        profile.publish_baseline()


def served_version() -> str:
    """Registry version this instance serves, for labels and responses."""
    return state["info"]["version"] if state["info"] else "unknown"


def sync_champion() -> bool:
    """Follow the registry: reload when the alias in MODEL_URI points at another version.

    Returns True if the model changed. Does nothing for an instance pinned to a version.
    """
    parsed = parse_model_uri(MODEL_URI)
    if parsed is None or is_pinned():
        return False
    import mlflow.pyfunc

    with promote_lock:
        current = str(MlflowClient().get_model_version_by_alias(parsed[0], parsed[1]).version)
        if state["info"] is not None and state["info"]["version"] == current:
            return False
        model = mlflow.pyfunc.load_model(f"models:/{parsed[0]}/{current}")
        set_served(model, load_model_info(MODEL_URI))
        MODEL_CHANGES.labels("sync").inc()
        logger.info("Following %s: now serving v%s", MODEL_URI, current)
        return True


def _follow_registry(stop: threading.Event) -> None:
    while not stop.wait(MODEL_POLL_SECONDS):
        try:
            sync_champion()
        except Exception as exc:  # the registry may be briefly unreachable; keep serving what we have
            logger.warning("Registry sync failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Try to load the registered model; the API stays up without one."""
    model = info = None
    try:
        import mlflow.pyfunc

        model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Loaded model %s", MODEL_URI)
    except Exception as exc:  # model may not be trained/registered yet
        logger.warning("No model loaded (%s): %s", MODEL_URI, exc)
    try:
        info = load_model_info(MODEL_URI)
    except Exception as exc:  # the UI model page degrades gracefully
        logger.warning("No model info (%s): %s", MODEL_URI, exc)
    set_served(model, info)
    stop = threading.Event()
    if MODEL_POLL_SECONDS > 0 and not is_pinned():
        threading.Thread(target=_follow_registry, args=(stop,), daemon=True).start()
    yield
    stop.set()


# Behind the Kubernetes Ingress the API is served under /api (the prefix is stripped before it reaches us).
# ROOT_PATH=/api makes Swagger build its URLs with that prefix; unset when the API is reached directly.
app = FastAPI(
    title="Churn Prediction API", version="0.1.0", lifespan=lifespan, root_path=os.getenv("ROOT_PATH", "")
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One log line per request: method, path, status, duration and model version.

    No query string and no body, so no customer data reaches the logs. A caller's X-Request-ID is
    kept when it is plain text (it could otherwise forge log lines); the id is echoed in the response.
    """
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("%s %s failed with an unhandled error", request.method, request.url.path)
            raise
        if request.url.path in QUIET_PATHS:
            level = logging.DEBUG
        else:
            level = logging.ERROR if response.status_code >= 500 else logging.INFO
        logger.log(
            level,
            "%s %s -> %d in %.1f ms (model v%s)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
            served_version(),
        )
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe; reports whether a model is loaded."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness probe: 200 only once a model is loaded, so a rollout never routes traffic to an empty pod."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": served_version()}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(RequestValidationError)
async def count_validation_errors(request: Request, exc: RequestValidationError):
    """Count rejected requests (422) so monitoring sees bad input, then answer as FastAPI would."""
    REQUESTS.labels(request.url.path.removeprefix("/v1/"), "422", served_version()).inc()
    return await request_validation_exception_handler(request, exc)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(features: CustomerFeatures) -> PredictResponse:
    """Return the calibrated churn probability for one customer.

    Missing fields, wrong types and out-of-range numbers are rejected with 422. A category never
    seen in training is still scored, with a warning in the response.
    """
    start = time.perf_counter()
    version = served_version()
    try:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        frame = pd.DataFrame([features.model_dump()]).astype({"TotalCharges": "float64"})
        prob = float(state["model"].predict(frame)[0])
        unknown = unknown_categories(features)
        for field, _ in unknown:
            UNKNOWN_CATEGORY.labels(field).inc()
        if state["profile"] is not None:
            try:
                state["profile"].observe(features.model_dump())
            except Exception as exc:  # monitoring must never break a prediction
                logger.warning("Could not record feature drift metrics: %s", exc)
        PREDICTIONS.labels("churn" if prob >= 0.5 else "stay", version).inc()
        CHURN_PROBABILITY.labels(version).observe(prob)
        REQUESTS.labels("predict", "200", version).inc()
        return PredictResponse(
            churn_probability=prob,
            model_version=version if version != "unknown" else None,
            warnings=[f"{field}: unseen category {value!r}" for field, value in unknown],
        )
    except HTTPException as exc:
        REQUESTS.labels("predict", str(exc.status_code), version).inc()
        raise
    except Exception as exc:
        REQUESTS.labels("predict", "500", version).inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.labels("predict", version).observe(time.perf_counter() - start)


@app.get("/v1/model")
def model_info() -> dict[str, Any]:
    """Registry version, metrics, selected configuration, provenance and curves of the served model."""
    if state["info"] is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    return state["info"]


@app.get("/v1/model/figures/{name}")
def model_figure(name: str) -> Response:
    """PNG figure (calibration or profit curve) logged by the training run of the served model."""
    if name not in FIGURES:
        raise HTTPException(status_code=404, detail="Unknown figure")
    if state["info"] is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    if name not in state["figures"]:
        state["figures"][name] = load_figure(state["info"]["run_id"], name)
    return Response(state["figures"][name], media_type="image/png")


@app.get("/v1/model/challenger")
def challenger_info() -> dict[str, Any]:
    """The model waiting for approval (alias `challenger`), described like the served one."""
    try:
        return load_model_info(f"models:/{MODEL_NAME}@{CHALLENGER}")
    except MlflowException as exc:
        raise HTTPException(status_code=404, detail="No model is waiting for approval") from exc


def _deny(action: str, reason: str, status: int, detail: str) -> HTTPException:
    """Count and log a refused admin call (never the key itself), then return the error to raise."""
    ADMIN_ACTIONS.labels(action, "denied").inc()
    logger.warning("admin action=%s denied: %s", action, reason)
    return HTTPException(status_code=status, detail=detail)


@contextmanager
def _audited(action: str):
    """Count and log an admin call that passed the key check but could not be carried out."""
    try:
        yield
    except HTTPException as exc:
        ADMIN_ACTIONS.labels(action, "failed").inc()
        logger.warning("admin action=%s failed: %s", action, exc.detail)
        raise


def _require_admin(key: str | None, action: str) -> None:
    if not ADMIN_KEY:
        raise _deny(action, "ADMIN_KEY is not configured", 503, "Approval is disabled: ADMIN_KEY is not configured")
    if not key or not hmac.compare_digest(key.encode(), ADMIN_KEY.encode()):
        raise _deny(action, "invalid admin key", 401, "Invalid admin key")
    if is_pinned():
        raise _deny(
            action,
            f"instance is pinned to {MODEL_URI}",
            409,
            f"This instance is pinned to {MODEL_URI}; deploy another version instead",
        )


def _activate(client: MlflowClient, version: str, kind: str, tag: str) -> dict[str, Any]:
    """Serve `version` and make it the champion. The registry is only touched after the model proved usable.

    The model must load and score a sample customer with the current API schema; otherwise
    nothing changes. Caller holds `promote_lock`.
    """
    import mlflow.pyfunc

    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
        info = load_model_info(f"models:/{MODEL_NAME}/{version}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"v{version} cannot be loaded, nothing changed: {exc}") from exc
    try:
        model.predict(pd.DataFrame([EXAMPLE]).astype({"TotalCharges": "float64"}))
    except Exception as exc:
        detail = f"v{version} does not accept the current input schema, nothing changed: {exc}"
        raise HTTPException(status_code=422, detail=detail) from exc
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    try:
        if str(client.get_model_version_by_alias(MODEL_NAME, CHALLENGER).version) == version:
            client.delete_registered_model_alias(MODEL_NAME, CHALLENGER)
    except MlflowException:
        pass  # no challenger waiting
    client.set_model_version_tag(MODEL_NAME, version, tag, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    set_served(model, {**info, "alias": CHAMPION})
    MODEL_CHANGES.labels(kind).inc()
    ADMIN_ACTIONS.labels(kind, "ok").inc()
    logger.info("admin action=%s ok: v%s is now served as %s", kind, version, CHAMPION)
    return state["info"]


@app.get("/v1/model/versions")
def model_versions() -> list[dict[str, Any]]:
    """All registered versions, newest first, with metrics, aliases and approval history."""
    try:
        return list_versions(MODEL_NAME)
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail=f"Registry unavailable: {exc}") from exc


@app.post("/v1/model/promote")
def approve_challenger(body: VersionRequest, x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Approve the challenger: make it the champion and start serving it.

    Needs the admin key. `version` must still be the current challenger, so an approval always
    applies to the model the approver was looking at.
    """
    _require_admin(x_admin_key, "approve")
    with _audited("approve"), promote_lock:
        client = MlflowClient()
        try:
            current = client.get_model_version_by_alias(MODEL_NAME, CHALLENGER)
        except MlflowException as exc:
            raise HTTPException(status_code=404, detail="No model is waiting for approval") from exc
        if str(current.version) != body.version:
            raise HTTPException(
                status_code=409,
                detail=f"The challenger is now v{current.version}, not v{body.version}; reload the page",
            )
        return _activate(client, body.version, "approve", "approved_at")


@app.post("/v1/model/rollback")
def restore_version(body: VersionRequest, x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Serve an earlier registered version again by pointing the champion alias at it. Needs the admin key."""
    _require_admin(x_admin_key, "rollback")
    with _audited("rollback"), promote_lock:
        client = MlflowClient()
        try:
            client.get_model_version(MODEL_NAME, body.version)
        except MlflowException as exc:
            raise HTTPException(status_code=404, detail=f"Version {body.version} does not exist") from exc
        if state["info"] is not None and state["info"]["version"] == body.version:
            raise HTTPException(status_code=409, detail=f"v{body.version} is already the champion")
        return _activate(client, body.version, "rollback", "restored_at")
