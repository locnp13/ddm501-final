"""Tests for the contact-budget metrics."""
import numpy as np
import pandas as pd

from src.training.business import best_row, profit_curve, selection_gap, top_k_mask

ECON = {"value_months": 12, "retention_cost_rate": 0.10, "retention_success": 0.30, "k_grid": [0.25, 0.5, 1.0]}


def test_top_k_mask_picks_highest_scores() -> None:
    assert top_k_mask(np.array([0.1, 0.9, 0.5, 0.7]), 0.5).tolist() == [False, True, False, True]


def test_profit_curve_hand_computed() -> None:
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.3, 0.1])
    monthly = np.array([100.0, 100.0, 50.0, 50.0])
    curve = profit_curve(y, score, monthly, ECON).set_index("k")
    # k=0.25: contact customer 0 -> saved 0.3*1200 = 360, cost 0.1*1200 = 120
    assert curve.loc[0.25, "profit"] == 240.0
    assert curve.loc[0.25, "recall"] == 0.5
    # k=0.5: adds a non-churner costing 120
    assert curve.loc[0.5, "profit"] == 120.0
    # k=1.0: all four; saved 360 + 180, cost 120 + 120 + 60 + 60
    assert curve.loc[1.0, "profit"] == 540.0 - 360.0
    assert best_row(curve.reset_index())["k"] == 0.25


def test_selection_gap() -> None:
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
    group = pd.Series(["a", "a", "b", "b", "b", "b", "b", "b", "b", "b"])
    # top 10% = 1 customer (in group a): rates a=0.5, b=0.0
    assert selection_gap(score, group, 0.10) == 0.5
