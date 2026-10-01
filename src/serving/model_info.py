"""Describe the served model for the UI: registry version, metrics, choices, provenance and curves."""
import io
import tempfile
from pathlib import Path
from typing import Any

import mlflow.artifacts
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

FIGURES = ("calibration.png", "profit_curve.png")
METRIC_KEYS = (
    "pr_auc", "roc_auc", "brier", "brier_uncalibrated", "best_k", "best_profit", "precision", "recall", "f1",
    "baseline_pr_auc", "selected_cv_pr_auc", "dpd_gender", "dpd_senior",
)
ECONOMICS_KEYS = ("value_months", "retention_cost_rate", "retention_success")


def parse_model_uri(uri: str) -> tuple[str, str | None, str | None] | None:
    """Split `models:/name@alias` or `models:/name/<version>` into (name, alias, version).

    Exactly one of alias and version is set. Other forms (such as stages) are not described.
    """
    if not uri.startswith("models:/"):
        return None
    rest = uri.removeprefix("models:/")
    if "@" in rest:
        name, alias = rest.split("@", 1)
        return name, alias, None
    name, _, version = rest.partition("/")
    return (name, None, version) if version.isdigit() else None


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
    parsed = parse_model_uri(uri)
    if parsed is None:
        return None
    name, alias, pinned_version = parsed
    client = MlflowClient()
    version = (
        client.get_model_version_by_alias(name, alias) if alias else client.get_model_version(name, pinned_version)
    )
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
        "version": str(version.version),
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


def list_versions(name: str) -> list[dict[str, Any]]:
    """Every registered version, newest first, with its headline metrics, aliases and approval history."""
    client = MlflowClient()
    # search_model_versions leaves `aliases` empty, so read them from the registered model.
    aliases: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        aliases.setdefault(str(version), []).append(alias)
    rows = []
    for v in client.search_model_versions(f"name='{name}'"):
        try:
            run = client.get_run(v.run_id)
            metrics, params = run.data.metrics, run.data.params
        except MlflowException:  # the run was deleted; the version still exists
            metrics, params = {}, {}
        rows.append(
            {
                "version": str(v.version),
                "aliases": sorted(aliases.get(str(v.version), [])),
                "created_at": v.creation_timestamp,
                "approved_at": v.tags.get("approved_at"),
                "restored_at": v.tags.get("restored_at"),
                "algorithm": params.get("selected.model"),
                **{k: metrics.get(k) for k in ("pr_auc", "roc_auc", "brier", "best_profit")},
            }
        )
    return sorted(rows, key=lambda r: -int(r["version"]))


def load_figure(run_id: str, name: str) -> bytes:
    """PNG artifact of the run that produced the model; `name` must be one of FIGURES."""
    if name not in FIGURES:
        raise KeyError(name)
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name, dst_path=tmp)
        return Path(path).read_bytes()
