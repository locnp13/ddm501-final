"""Model-validation tests for the three-part quality gate."""
import pytest

from src.training.registry import passes_gate

GATE = {"min_pr_auc": 0.50, "baseline_margin": 0.02, "champion_tolerance": 0.005}


@pytest.mark.parametrize(
    ("pr_auc", "baseline", "champion", "expected"),
    [
        (0.66, 0.62, None, True),  # first run: no champion yet
        (0.66, 0.62, 0.66, True),  # equal to champion
        (0.66, 0.62, 0.664, True),  # within tolerance of champion
        (0.66, 0.62, 0.67, False),  # regressed against champion
        (0.63, 0.62, None, False),  # does not beat baseline by the margin
        (0.49, 0.30, None, False),  # below the absolute floor
    ],
)
def test_gate_decision(pr_auc: float, baseline: float, champion: float | None, expected: bool) -> None:
    passed, reasons = passes_gate(pr_auc, baseline, champion, GATE)
    assert passed is expected
    assert bool(reasons) is not expected
