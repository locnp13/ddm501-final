"""Business metrics: contact-budget curve (Recall@k, expected profit) and group selection gaps."""
import numpy as np
import pandas as pd


def top_k_mask(score: np.ndarray, k_frac: float) -> np.ndarray:
    """Boolean mask of the top `k_frac` share of customers by score (at least one)."""
    n = max(1, int(round(len(score) * k_frac)))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-np.asarray(score), kind="stable")[:n]] = True
    return mask


def profit_curve(y: np.ndarray, score: np.ndarray, monthly: np.ndarray, econ: dict) -> pd.DataFrame:
    """Realised profit, recall and precision when contacting the top-k share of customers.

    A contacted customer costs `retention_cost_rate` of their yearly revenue; if they would
    have churned, a `retention_success` share of that yearly revenue is saved.
    """
    y, monthly = np.asarray(y), np.asarray(monthly, dtype=float)
    value = monthly * econ["value_months"]
    cost = econ["retention_cost_rate"] * value
    saved = econ["retention_success"] * value * y
    rows = []
    for k in econ["k_grid"]:
        m = top_k_mask(score, k)
        caught = int(y[m].sum())
        rows.append(
            {
                "k": k,
                "contacted": int(m.sum()),
                "recall": caught / max(1, int(y.sum())),
                "precision": caught / int(m.sum()),
                "profit": float(saved[m].sum() - cost[m].sum()),
            }
        )
    return pd.DataFrame(rows)


def best_row(curve: pd.DataFrame) -> pd.Series:
    """The k with the highest profit."""
    return curve.loc[curve["profit"].idxmax()]


def selection_gap(score: np.ndarray, group: pd.Series, k_frac: float = 0.10) -> float:
    """Largest difference in top-k selection rate between groups (demographic-parity gap)."""
    selected = pd.Series(top_k_mask(score, k_frac), index=group.index)
    rates = selected.groupby(group).mean()
    return float(rates.max() - rates.min())
