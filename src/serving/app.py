"""Churn prediction API: health, Prometheus metrics, and model-backed prediction."""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.serving.model_info import FIGURES, load_figure, load_model_info

logger = logging.getLogger("churn-api")

MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")

REQUESTS = Counter("churn_requests_total", "Requests by endpoint and status", ["endpoint", "status"])
LATENCY = Histogram("churn_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("churn_predictions_total", "Predictions by class", ["label"])
CHURN_PROBABILITY = Histogram(
    "churn_probability", "Predicted churn probability", buckets=[i / 10 for i in range(11)]
)
MODEL_LOADED = Gauge("churn_model_loaded", "1 if a model is loaded, else 0")

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


@app.post("/v1/predict")
def predict(features: dict[str, Any]) -> dict[str, Any]:
    """Return churn probability for one customer's feature dict."""
    start = time.perf_counter()
    try:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        result = state["model"].predict(pd.DataFrame([features]))
        prob = float(result[0])
        PREDICTIONS.labels("churn" if prob >= 0.5 else "stay").inc()
        CHURN_PROBABILITY.observe(prob)
        REQUESTS.labels("predict", "200").inc()
        return {"churn_probability": prob}
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
