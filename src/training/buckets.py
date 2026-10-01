"""Bucketing of input values for drift monitoring. Standard library only, so the API image can import it.

Numeric inputs get quantile buckets, categorical inputs one bucket per value seen in training, and
anything unseen at serving time falls into `other`. The baseline itself is built in baseline.py.
"""
import bisect
import math
from typing import Any

BASELINE_FILE = "feature_baseline.json"
NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges")
EPSILON = 1e-4  # share floor, so an empty bucket does not make PSI infinite
MISSING = "missing"
OTHER = "other"


def numeric_labels(cuts: list[float]) -> list[str]:
    """Readable, sortable bucket names for the intervals defined by `cuts`."""
    labels = [f"00_<{cuts[0]:g}"]
    labels += [f"{i:02d}_{cuts[i - 1]:g}-{cuts[i]:g}" for i in range(1, len(cuts))]
    labels.append(f"{len(cuts):02d}_>={cuts[-1]:g}")
    return labels


def bucket_label(spec: dict[str, Any], value: Any) -> str:
    """The bucket a raw input value falls into; the label set is bounded by the baseline."""
    if spec["type"] == "numeric":
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return MISSING
        try:
            number = float(value)
        except (TypeError, ValueError):
            return MISSING
        return numeric_labels(spec["cuts"])[bisect.bisect_right(spec["cuts"], number)]
    label = str(value)
    return label if label in spec["shares"] else OTHER
