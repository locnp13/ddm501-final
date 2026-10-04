"""Input drift metrics: count every request into the training buckets and publish the training shares.

Prometheus turns the two into a PSI per feature (k8s/monitoring/prometheus/alerts.yml). Only bucket
counts are exported, never the values of a customer. The label sets are bounded: numeric inputs use
the baseline's quantile buckets, categorical inputs the categories seen in training plus `other`.
"""
import logging
from typing import Any

import mlflow.artifacts
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge

from src.training.buckets import BASELINE_FILE, EPSILON, OTHER, bucket_label

logger = logging.getLogger("churn-api")

FEATURE_VALUES = Counter(
    "churn_feature_values_total",
    "Requests per input feature bucket and model version",
    ["model_version", "feature", "bucket"],
)
FEATURE_BASELINE = Gauge(
    "churn_feature_baseline_share",
    "Share of each input feature bucket in the training data of the served model",
    ["model_version", "feature", "bucket"],
)


class FeatureProfile:
    """Training distribution of the served model, used to bucket live requests."""

    def __init__(self, baseline: dict[str, Any], version: str) -> None:
        self.features: dict[str, Any] = baseline["features"]
        self.version = version

    def publish_baseline(self) -> None:
        """Expose the training shares and start every bucket counter at 0.

        Empty buckets get a small floor so PSI stays finite. The counters are created up front
        because Prometheus' increase() cannot see a series' first jump from nothing to N: a burst
        that arrives before the first scrape of a new series would be missed and PSI would be garbage.
        """
        for feature, spec in self.features.items():
            buckets = list(spec["shares"])
            if spec["type"] == "categorical":
                buckets.append(OTHER)
            for bucket in buckets:
                FEATURE_BASELINE.labels(self.version, feature, bucket).set(max(spec["shares"].get(bucket, 0), EPSILON))
                FEATURE_VALUES.labels(self.version, feature, bucket)  # touching it creates the child at 0

    def observe(self, row: dict[str, Any]) -> None:
        """Count one request into the bucket of each feature."""
        for feature, spec in self.features.items():
            FEATURE_VALUES.labels(self.version, feature, bucket_label(spec, row.get(feature))).inc()


def load_profile(info: dict[str, Any] | None) -> FeatureProfile | None:
    """The profile of the model described by `info`, or None when it has no baseline (older models)."""
    if info is None:
        return None
    try:
        # MLflow answers 500 for a missing artifact and the client retries for about a minute, so look first.
        if BASELINE_FILE not in {a.path for a in MlflowClient().list_artifacts(info["run_id"])}:
            logger.info("v%s was trained without a feature baseline, input drift is not monitored", info["version"])
            return None
        baseline = mlflow.artifacts.load_dict(f"runs:/{info['run_id']}/{BASELINE_FILE}")
    except Exception as exc:
        logger.warning("No feature baseline for v%s, input drift is not monitored: %s", info["version"], exc)
        return None
    return FeatureProfile(baseline, info["version"])
