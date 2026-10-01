"""Describe the served model for the UI: registry version, metrics, choices, provenance and curves."""
import io
import tempfile
from pathlib import Path
from typing import Any

import mlflow.artifacts
import pandas as pd
from mlflow.tracking import MlflowClient

FIGURES = ("calibration.png", "profit_curve.png")
METRIC_KEYS = (
    "pr_auc", "roc_auc", "brier", "brier_uncalibrated", "best_k", "best_profit", "precision", "recall", "f1",
    "baseline_pr_auc", "selected_cv_pr_auc", "dpd_gender", "dpd_senior",
)
ECONOMICS_KEYS = ("value_months", "retention_cost_rate", "retention_success")


def parse_alias_uri(uri: str) -> tuple[str, str] | None:
    """Split `models:/name@alias` into (name, alias); other URI forms are not described."""
    if not uri.startswith("models:/") or "@" not in uri:
        return None
    name, alias = uri.removeprefix("models:/").split("@", 1)
    return name, alias


def _number(value: str) -> Any:
    """MLflow stores params as strings; turn numbers and booleans back into their types."""
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except (TypeError, ValueError):
            continue
    return value


def load_model_info(uri: str) -> dict[str, Any] | None:
    """Collect what the UI shows about the model behind `uri`, or None if it cannot be described."""
    parsed = parse_alias_uri(uri)
    if parsed is None:
        return None
    name, alias = parsed
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)
    params = run.data.params

    def group(prefix: str) -> dict[str, Any]:
        return {k.removeprefix(prefix): _number(v) for k, v in params.items() if k.startswith(prefix)}

    curve = mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/profit_curve.csv")
    candidates = client.search_runs(
        [run.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'", max_results=100
    )
    return {
        "name": name,
        "alias": alias,
        "version": version.version,
        "run_id": run.info.run_id,
        "created_at": version.creation_timestamp,
        "metrics": {k: run.data.metrics[k] for k in METRIC_KEYS if k in run.data.metrics},
        "selected": group("selected."),
        "best_params": group("best."),
        "economics": {k: v for k, v in group("economics.").items() if k in ECONOMICS_KEYS},
        "provenance": {k: run.data.tags.get(k, "unknown") for k in ("git_commit", "git_dirty", "data_md5")},
        "profit_curve": pd.read_csv(io.StringIO(curve)).to_dict(orient="records"),
        "candidates": sorted(
            (
                {"name": c.data.tags.get("mlflow.runName", ""), "cv_pr_auc": c.data.metrics["cv_pr_auc"]}
                for c in candidates
                if "cv_pr_auc" in c.data.metrics and "-fe" in c.data.tags.get("mlflow.runName", "")
            ),
            key=lambda c: -c["cv_pr_auc"],
        ),
    }


def load_figure(run_id: str, name: str) -> bytes:
    """PNG artifact of the run that produced the model; `name` must be one of FIGURES."""
    if name not in FIGURES:
        raise KeyError(name)
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name, dst_path=tmp)
        return Path(path).read_bytes()
