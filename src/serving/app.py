"""Churn prediction API: health, Prometheus metrics, and model-backed prediction."""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.serving.model_info import FIGURES, load_figure, load_model_info
from src.serving.schemas import CustomerFeatures, PredictResponse, unknown_categories

logger = logging.getLogger("churn-api")

MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")

REQUESTS = Counter("churn_requests_total", "Requests by endpoint and status", ["endpoint", "status"])
LATENCY = Histogram("churn_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("churn_predictions_total", "Predictions by class", ["label"])
CHURN_PROBABILITY = Histogram(
    "churn_probability", "Predicted churn probability", buckets=[i / 10 for i in range(11)]
)
MODEL_LOADED = Gauge("churn_model_loaded", "1 if a model is loaded, else 0")
UNKNOWN_CATEGORY = Counter(
    "churn_unknown_category_total", "Requests with a category never seen in training", ["field"]
)

state: dict[str, Any] = {"model": None, "info": None, "figures": {}}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Try to load the registered model; the API stays up without one."""
    try:
        import mlflow.pyfunc

        state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Loaded model %s", MODEL_URI)
    except Exception as exc:  # model may not be trained/registered yet
        logger.warning("No model loaded (%s): %s", MODEL_URI, exc)
    try:
        state["info"] = load_model_info(MODEL_URI)
    except Exception as exc:  # the UI model page degrades gracefully
        logger.warning("No model info (%s): %s", MODEL_URI, exc)
    MODEL_LOADED.set(1 if state["model"] is not None else 0)
    yield


app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe; reports whether a model is loaded."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(RequestValidationError)
async def count_validation_errors(request: Request, exc: RequestValidationError):
    """Count rejected requests (422) so monitoring sees bad input, then answer as FastAPI would."""
    REQUESTS.labels(request.url.path.removeprefix("/v1/"), "422").inc()
    return await request_validation_exception_handler(request, exc)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(features: CustomerFeatures) -> PredictResponse:
    """Return the calibrated churn probability for one customer.

    Missing fields, wrong types and out-of-range numbers are rejected with 422. A category never
    seen in training is still scored, with a warning in the response.
    """
    start = time.perf_counter()
    try:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        frame = pd.DataFrame([features.model_dump()]).astype({"TotalCharges": "float64"})
        prob = float(state["model"].predict(frame)[0])
        unknown = unknown_categories(features)
        for field, _ in unknown:
            UNKNOWN_CATEGORY.labels(field).inc()
        PREDICTIONS.labels("churn" if prob >= 0.5 else "stay").inc()
        CHURN_PROBABILITY.observe(prob)
        REQUESTS.labels("predict", "200").inc()
        version = state["info"]["version"] if state["info"] else None
        return PredictResponse(
            churn_probability=prob,
            model_version=version,
            warnings=[f"{field}: unseen category {value!r}" for field, value in unknown],
        )
    except HTTPException as exc:
        REQUESTS.labels("predict", str(exc.status_code)).inc()
        raise
    except Exception as exc:
        REQUESTS.labels("predict", "500").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.labels("predict").observe(time.perf_counter() - start)


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
