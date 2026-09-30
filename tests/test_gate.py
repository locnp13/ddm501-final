"""Model-validation tests for the quality gate."""
import pytest

from src.training.train import passes_gate


@pytest.mark.parametrize(
    ("pr_auc", "threshold", "expected"),
    [(0.62, 0.50, True), (0.50, 0.50, True), (0.49, 0.50, False)],
)
def test_gate_decision(pr_auc: float, threshold: float, expected: bool) -> None:
    assert passes_gate(pr_auc, threshold) is expected
