"""Model registry helpers: quality gate, champion lookup, aliases and run provenance."""
import os
from pathlib import Path

import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "churn-model"
CHALLENGER = "challenger"
CHAMPION = "champion"
DVC_LOCK = Path("dvc.lock")
PROCESSED_KEY = "data/processed/churn.csv"


def passes_gate(
    pr_auc: float, baseline_pr_auc: float | None, champion_pr_auc: float | None, gate: dict
) -> tuple[bool, list[str]]:
    """Three-part quality gate; returns (passed, reasons for failure)."""
    reasons = []
    if pr_auc < gate["min_pr_auc"]:
        reasons.append(f"PR-AUC {pr_auc:.3f} below floor {gate['min_pr_auc']}")
    if baseline_pr_auc is not None and pr_auc < baseline_pr_auc + gate["baseline_margin"]:
        reasons.append(f"PR-AUC {pr_auc:.3f} does not beat baseline {baseline_pr_auc:.3f} + {gate['baseline_margin']}")
    if champion_pr_auc is not None and pr_auc < champion_pr_auc - gate["champion_tolerance"]:
        reasons.append(f"PR-AUC {pr_auc:.3f} worse than champion {champion_pr_auc:.3f} - {gate['champion_tolerance']}")
    return not reasons, reasons


def champion_pr_auc(client: MlflowClient) -> float | None:
    """Test PR-AUC of the model behind the `champion` alias, or None if there is no champion."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get("pr_auc")


def provenance(lock_path: Path = DVC_LOCK) -> dict[str, str]:
    """Git commit and data hash to tag a run with, so it can be reproduced."""
    md5 = "unknown"
    if lock_path.exists():
        lock = yaml.safe_load(lock_path.read_text()) or {}
        for out in lock.get("stages", {}).get("ingest", {}).get("outs", []):
            if out.get("path") == PROCESSED_KEY:
                md5 = out.get("md5", "unknown")
    return {
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "git_dirty": os.getenv("GIT_DIRTY", "unknown"),
        "data_md5": md5,
    }


def promote(client: MlflowClient, version: str | None = None) -> str:
    """Point `champion` at `version` (default: the current challenger) and clear `challenger`."""
    if version is None:
        version = client.get_model_version_by_alias(MODEL_NAME, CHALLENGER).version
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    try:
        client.delete_registered_model_alias(MODEL_NAME, CHALLENGER)
    except MlflowException:
        pass
    return str(version)
