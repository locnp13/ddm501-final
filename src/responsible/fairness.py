"""Fairness of the contact decision, and a post-processing mitigation (equal opportunity).

The model itself only scores customers; the decision that affects people is who gets contacted with a
retention offer (the top-k share by score, k chosen for profit, see src/training/business.py). Fairness
is therefore measured on that decision, per group of a sensitive attribute:

- selection rate: share of the group that is contacted (demographic parity compares these);
- true positive rate: share of the group's churners that are contacted (equal opportunity compares these);
- false positive rate, precision, and calibration (mean predicted vs actual churn) per group.

Mitigation: group-specific thresholds chosen so that each group's churners are contacted at the same rate,
under the same overall contact budget (Hardt et al., 2016). Thresholds are fitted on one half of the test
set and evaluated on the other half, repeated over many random splits, because the groups are small.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.business import top_k_mask


def group_metrics(y: np.ndarray, selected: np.ndarray, score: np.ndarray, group: pd.Series) -> pd.DataFrame:
    """Per-group size, actual churn rate, selection rate, TPR, FPR, precision and calibration."""
    frame = pd.DataFrame({"y": np.asarray(y), "sel": np.asarray(selected, dtype=bool),
                          "score": np.asarray(score, dtype=float), "group": np.asarray(group)})
    rows = []
    for name, g in frame.groupby("group"):
        positives, negatives = g["y"] == 1, g["y"] == 0
        rows.append(
            {
                "group": name,
                "n": len(g),
                "churn_rate": g["y"].mean(),
                "mean_score": g["score"].mean(),
                "selection_rate": g["sel"].mean(),
                "tpr": g.loc[positives, "sel"].mean() if positives.any() else np.nan,
                "fpr": g.loc[negatives, "sel"].mean() if negatives.any() else np.nan,
                "precision": g.loc[g["sel"], "y"].mean() if g["sel"].any() else np.nan,
                "brier": ((g["score"] - g["y"]) ** 2).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def disparities(metrics: pd.DataFrame) -> dict[str, float]:
    """Gaps between groups: largest minus smallest, and the disparate-impact ratio of selection rates."""
    sel = metrics["selection_rate"]
    return {
        "demographic_parity_diff": float(sel.max() - sel.min()),
        "disparate_impact_ratio": float(sel.min() / sel.max()) if sel.max() > 0 else float("nan"),
        "equal_opportunity_diff": float(metrics["tpr"].max() - metrics["tpr"].min()),
        "fpr_diff": float(metrics["fpr"].max() - metrics["fpr"].min()),
        "calibration_gap_diff": float((metrics["mean_score"] - metrics["churn_rate"]).abs().max()),
    }


def equal_opportunity_thresholds(
    score: np.ndarray, y: np.ndarray, group: pd.Series, budget: float, grid: int = 400
) -> dict:
    """Per-group thresholds giving every group's churners the same contact rate, using `budget` of customers.

    For each target TPR on a grid, each group's threshold is the score of its churner at that rank; the
    target whose total number of contacts is closest to the budget wins.
    """
    score, y, group = np.asarray(score, dtype=float), np.asarray(y), np.asarray(group)
    target_contacts = budget * len(score)
    best, best_gap = None, np.inf
    for tpr in np.linspace(0.01, 1.0, grid):
        thresholds, contacts = {}, 0
        for name in np.unique(group):
            in_group = group == name
            pos_scores = np.sort(score[in_group & (y == 1)])[::-1]
            if len(pos_scores) == 0:
                thresholds[name] = np.inf
                continue
            rank = max(1, int(np.ceil(tpr * len(pos_scores))))
            thresholds[name] = pos_scores[rank - 1]
            contacts += int((score[in_group] >= thresholds[name]).sum())
        gap = abs(contacts - target_contacts)
        if gap < best_gap:
            best, best_gap = thresholds, gap
    return best


def apply_thresholds(score: np.ndarray, group: pd.Series, thresholds: dict) -> np.ndarray:
    """Contact decision with a threshold per group (unknown groups are never contacted)."""
    group = np.asarray(group)
    return np.array([s >= thresholds.get(g, np.inf) for s, g in zip(np.asarray(score), group)])


def profit(y: np.ndarray, selected: np.ndarray, monthly: np.ndarray, econ: dict) -> float:
    """Realised profit of a contact decision, with the same assumptions as business.profit_curve."""
    value = np.asarray(monthly, dtype=float) * econ["value_months"]
    selected = np.asarray(selected, dtype=bool)
    gain = econ["retention_success"] * value * np.asarray(y) - econ["retention_cost_rate"] * value
    return float(gain[selected].sum())


def mitigation_study(
    score: np.ndarray, y: np.ndarray, group: pd.Series, monthly: np.ndarray, budget: float, econ: dict,
    repeats: int = 50, seed: int = 0,
) -> pd.DataFrame:
    """Baseline top-k versus equal-opportunity thresholds, fitted and evaluated on disjoint halves.

    Returns one row per repeat with the TPR gap, selection gap, contact share and profit of both decisions
    on the evaluation half.
    """
    score, y, monthly = np.asarray(score, dtype=float), np.asarray(y), np.asarray(monthly, dtype=float)
    group = pd.Series(np.asarray(group))
    strata = group.astype(str) + "_" + pd.Series(y).astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(repeats):
        fit = np.zeros(len(y), dtype=bool)
        for _, idx in strata.groupby(strata).groups.items():
            idx = np.asarray(list(idx))
            fit[rng.choice(idx, size=len(idx) // 2, replace=False)] = True
        ev = ~fit
        thresholds = equal_opportunity_thresholds(score[fit], y[fit], group[fit], budget)
        base_sel = top_k_mask(score[ev], budget)
        fair_sel = apply_thresholds(score[ev], group[ev], thresholds)
        for name, sel in (("baseline", base_sel), ("equal_opportunity", fair_sel)):
            gaps = disparities(group_metrics(y[ev], sel, score[ev], group[ev]))
            rows.append(
                {
                    "repeat": r,
                    "decision": name,
                    "tpr_gap": gaps["equal_opportunity_diff"],
                    "selection_gap": gaps["demographic_parity_diff"],
                    "contact_share": float(np.mean(sel)),
                    "profit": profit(y[ev], sel, monthly[ev], econ),
                }
            )
    return pd.DataFrame(rows)
