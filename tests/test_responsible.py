"""Tests for the responsible-AI report: explanations (SHAP, LIME), fairness metrics and mitigation."""
import json

import numpy as np
import pandas as pd
import pytest

from src.responsible import report
from src.responsible.explain import RawEncoder, lime_explain, lime_explainer, lime_features, shap_explain, top_agreement
from src.responsible.fairness import (
    apply_thresholds,
    disparities,
    equal_opportunity_thresholds,
    group_metrics,
    mitigation_study,
    profit,
)
from src.training.business import profit_curve, top_k_mask
from src.training.data import clean, coerce_total_charges, validate
from src.training.train import build_pipeline, make_estimator
from tests import test_api
from tests.conftest import synthetic_telco

champion_dir = test_api.champion_dir  # a trained champion in a temporary MLflow store
ECON = {"value_months": 12, "retention_cost_rate": 0.1, "retention_success": 0.3, "k_grid": [0.3]}


@pytest.fixture(scope="module")
def fitted():
    df = clean(validate(coerce_total_charges(synthetic_telco(400, seed=1))))
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    pipe = build_pipeline(make_estimator("logreg", 0)).fit(X, y)
    return pipe, X, y


def predict_of(pipe):
    return lambda frame: pipe.predict_proba(frame)[:, 1]


def test_encoder_round_trip_keeps_values(fitted) -> None:
    _, X, _ = fitted
    enc = RawEncoder(X)
    back = enc.decode(enc.encode(X))
    expected = X.assign(TotalCharges=X["TotalCharges"].fillna(0))
    pd.testing.assert_frame_equal(back, expected.reset_index(drop=True), check_dtype=False)
    assert enc.readable("Contract", 0) == enc.categories["Contract"][0]
    assert enc.readable("tenure", 12) == "12"


def test_unseen_category_is_encoded_as_minus_one_and_decoded_to_a_known_value(fitted) -> None:
    _, X, _ = fitted
    enc = RawEncoder(X)
    row = X.head(1).assign(PaymentMethod="Momo")
    code = enc.encode(row)[0, enc.columns.index("PaymentMethod")]
    assert code == -1
    assert enc.decode(enc.encode(row))["PaymentMethod"][0] in enc.categories["PaymentMethod"]


def test_shap_values_add_up_to_the_prediction(fitted) -> None:
    pipe, X, _ = fitted
    enc = RawEncoder(X)
    customers = X.sample(15, random_state=0)
    res = shap_explain(predict_of(pipe), enc, X.sample(20, random_state=1), customers)
    reconstructed = res.base_value + res.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, predict_of(pipe)(customers), atol=1e-6)
    assert set(res.importance().index[:3]) & {"Contract", "tenure"}  # synthetic churn depends on these
    assert list(res.local(0, top=3).index)[0].split(" = ")[0] in enc.columns
    assert set(res.by_category("Contract").index) <= set(enc.categories["Contract"])


def test_lime_explains_one_customer_and_overlaps_with_shap(fitted) -> None:
    pipe, X, _ = fitted
    enc = RawEncoder(X)
    lime = lime_explainer(enc, X, seed=0)
    customer = X.iloc[[0]]
    weights = lime_explain(lime, predict_of(pipe), enc, customer, top=5, samples=500)
    assert len(weights) == 5
    names = lime_features(lime, predict_of(pipe), enc, customer, top=5)
    assert len(names) == 5 and set(names) <= set(enc.columns)


def test_top_agreement() -> None:
    assert top_agreement(["a", "b", "c"], ["a", "x"]) == 0.5
    assert top_agreement(["a"], []) == 0.0


def test_group_metrics_and_gaps_on_a_hand_made_case() -> None:
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sel = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.3, 0.2, 0.1])
    group = pd.Series(["a"] * 4 + ["b"] * 4)
    m = group_metrics(y, sel, score, group)
    assert m.loc["a", "selection_rate"] == 0.75 and m.loc["b", "selection_rate"] == 0.25
    assert m.loc["a", "tpr"] == 1.0 and m.loc["b", "tpr"] == 0.5
    assert m.loc["a", "fpr"] == 0.5 and m.loc["b", "fpr"] == 0.0
    gaps = disparities(m)
    assert gaps["demographic_parity_diff"] == 0.5
    assert gaps["equal_opportunity_diff"] == 0.5
    assert gaps["disparate_impact_ratio"] == pytest.approx(1 / 3)


def test_equal_opportunity_thresholds_equalise_tpr_within_budget() -> None:
    rng = np.random.default_rng(0)
    group = pd.Series(rng.integers(0, 2, 2000))
    y = (rng.random(2000) < np.where(group == 1, 0.45, 0.2)).astype(int)
    score = np.clip(0.3 * y + 0.25 * group + rng.normal(0.2, 0.15, 2000), 0, 1)  # group 1 scored higher
    budget = 0.3
    base = group_metrics(y, top_k_mask(score, budget), score, group)
    th = equal_opportunity_thresholds(score, y, group, budget)
    sel = apply_thresholds(score, group, th)
    fair = group_metrics(y, sel, score, group)
    assert abs(sel.mean() - budget) < 0.01
    assert disparities(fair)["equal_opportunity_diff"] < 0.02 < disparities(base)["equal_opportunity_diff"]


def test_unknown_group_is_never_contacted() -> None:
    assert not apply_thresholds(np.array([0.9]), pd.Series(["z"]), {"a": 0.1}).any()


def test_profit_matches_the_business_curve() -> None:
    rng = np.random.default_rng(1)
    y, score, monthly = rng.integers(0, 2, 300), rng.random(300), rng.uniform(20, 100, 300)
    curve = profit_curve(y, score, monthly, ECON)
    assert profit(y, top_k_mask(score, 0.3), monthly, ECON) == pytest.approx(curve.loc[0, "profit"])


def test_mitigation_study_compares_both_decisions_at_the_same_budget() -> None:
    rng = np.random.default_rng(2)
    group = pd.Series(rng.integers(0, 2, 800))
    y = (rng.random(800) < 0.3).astype(int)
    score = np.clip(0.3 * y + 0.2 * group + rng.normal(0.2, 0.1, 800), 0, 1)
    study = mitigation_study(score, y, group, rng.uniform(20, 100, 800), 0.3, ECON, repeats=5)
    assert set(study["decision"]) == {"baseline", "equal_opportunity"} and len(study) == 10
    means = study.groupby("decision").mean(numeric_only=True)
    assert means.loc["equal_opportunity", "tpr_gap"] < means.loc["baseline", "tpr_gap"]
    assert abs(means.loc["equal_opportunity", "contact_share"] - 0.3) < 0.03


def test_report_runs_end_to_end_on_the_registered_champion(champion_dir, tmp_path, monkeypatch) -> None:
    data = tmp_path / "churn.csv"
    clean(validate(coerce_total_charges(synthetic_telco(300, seed=4)))).to_csv(data, index=False)
    monkeypatch.chdir(test_api.__file__.rsplit("/tests/", 1)[0])  # params.yaml lives at the repo root
    out = tmp_path / "rai"
    results = report.main(["--data", str(data), "--out", str(out), "--explain", "10", "--background", "10",
                           "--lime", "2", "--repeats", "3", "--log-to-run"])
    assert {"summary.md", "results.json", "shap_importance.png", "local_explanations.png",
            "fairness_senior.png"} <= {p.name for p in out.iterdir()}
    assert json.loads((out / "results.json").read_text())["model_version"] == "1"
    assert results["shap"]["max_additivity_error"] < 1e-6
    assert set(results["fairness"]) == {"gender", "SeniorCitizen"}
    assert "Báo cáo Responsible AI" in (out / "summary.md").read_text()
