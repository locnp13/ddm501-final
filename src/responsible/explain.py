"""Model explanations at the level of the 19 raw inputs, with two independent methods: SHAP and LIME.

Both methods treat the served model as a black box (raw customer columns in, calibrated churn
probability out), so the explanations use the same words as the API and the business: "Contract",
"tenure", not one-hot columns. Categorical inputs are integer-coded for the explainers and decoded
back to their text values before every model call.

- SHAP (permutation estimator, Shapley values): additive. For one customer, the base value plus the
  contributions equals the predicted probability exactly. Averaged absolute values give a global ranking.
- LIME: fits a small linear model around one customer on perturbed copies. Local only, and not
  additive, but a different method, so agreement with SHAP is evidence that an explanation is stable.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Predict = Callable[[pd.DataFrame], np.ndarray]
INTEGER_COLUMNS = ("SeniorCitizen", "tenure")


class RawEncoder:
    """Two-way mapping between raw customer frames and the float matrices the explainers need."""

    def __init__(self, X: pd.DataFrame) -> None:
        self.columns = list(X.columns)
        self.categorical = [c for c in self.columns if X[c].dtype == object]
        self.categories = {c: sorted(X[c].astype(str).unique()) for c in self.categorical}

    @property
    def categorical_indices(self) -> list[int]:
        return [self.columns.index(c) for c in self.categorical]

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """Categories become their index (unseen ones -1); missing numbers become 0 (TotalCharges of new customers)."""
        out = np.zeros((len(X), len(self.columns)))
        for j, column in enumerate(self.columns):
            if column in self.categories:
                lookup = {value: i for i, value in enumerate(self.categories[column])}
                out[:, j] = X[column].astype(str).map(lookup).fillna(-1).to_numpy()
            else:
                out[:, j] = pd.to_numeric(X[column], errors="coerce").fillna(0).to_numpy()
        return out

    def decode(self, matrix: np.ndarray) -> pd.DataFrame:
        """Rebuild a raw frame the model accepts; codes are rounded and clipped to a known category."""
        matrix = np.atleast_2d(matrix)
        data: dict[str, Any] = {}
        for j, column in enumerate(self.columns):
            if column in self.categories:
                names = np.array(self.categories[column], dtype=object)
                codes = np.clip(np.rint(matrix[:, j]).astype(int), 0, len(names) - 1)
                data[column] = names[codes]
            elif column in INTEGER_COLUMNS:
                data[column] = np.rint(matrix[:, j]).astype(int)
            else:
                data[column] = matrix[:, j].astype(float)
        return pd.DataFrame(data, columns=self.columns)

    def readable(self, column: str, code: float) -> str:
        """Text value of an encoded input, for labels in plots and reports."""
        if column in self.categories:
            return self.categories[column][int(np.clip(round(code), 0, len(self.categories[column]) - 1))]
        return f"{code:g}"


@dataclass
class ShapResult:
    values: np.ndarray  # (customers, features): contribution of each input to the probability
    base_value: float  # average prediction over the background customers
    data: pd.DataFrame  # the explained customers, raw values
    columns: list[str]

    def importance(self) -> pd.Series:
        """Mean absolute contribution per input, largest first."""
        return pd.Series(np.abs(self.values).mean(axis=0), index=self.columns).sort_values(ascending=False)

    def by_category(self, column: str) -> pd.Series:
        """Average contribution of each value of a categorical input (which values push risk up or down)."""
        j = self.columns.index(column)
        return pd.Series(self.values[:, j]).groupby(self.data[column].to_numpy()).mean().sort_values(ascending=False)

    def local(self, i: int, top: int = 8) -> pd.Series:
        """Largest contributions for customer `i`, labelled `input = value`."""
        row = self.data.iloc[i]
        labels = [f"{c} = {row[c]}" for c in self.columns]
        series = pd.Series(self.values[i], index=labels)
        return series.reindex(series.abs().sort_values(ascending=False).index)[:top]


def shap_explain(
    predict: Predict, encoder: RawEncoder, background: pd.DataFrame, customers: pd.DataFrame, seed: int = 0,
    max_evals: int | None = None,
) -> ShapResult:
    """Shapley values of `customers` against `background` with the permutation estimator (model-agnostic)."""
    import shap

    def f(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(predict(encoder.decode(matrix)), dtype=float)

    masker = shap.maskers.Independent(encoder.encode(background), max_samples=len(background))
    n_features = len(encoder.columns)
    explainer = shap.PermutationExplainer(f, masker, feature_names=encoder.columns, seed=seed)
    explanation = explainer(encoder.encode(customers), max_evals=max_evals or 4 * n_features + 1, silent=True)
    base = np.atleast_1d(explanation.base_values)
    return ShapResult(
        values=np.asarray(explanation.values),
        base_value=float(base[0]),
        data=customers.reset_index(drop=True),
        columns=encoder.columns,
    )


def lime_explainer(encoder: RawEncoder, training: pd.DataFrame, seed: int = 0):
    """LIME explainer that perturbs customers like the training data (categories sampled by frequency)."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        encoder.encode(training),
        feature_names=encoder.columns,
        categorical_features=encoder.categorical_indices,
        categorical_names={encoder.columns.index(c): v for c, v in encoder.categories.items()},
        class_names=["stay", "churn"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )


def lime_explain(
    explainer, predict: Predict, encoder: RawEncoder, customer: pd.DataFrame, top: int = 8, samples: int = 2000
) -> pd.Series:
    """LIME weights for one customer: positive pushes towards churn. Labels are LIME's conditions."""

    def proba(matrix: np.ndarray) -> np.ndarray:
        p = np.asarray(predict(encoder.decode(matrix)), dtype=float)
        return np.column_stack([1 - p, p])

    exp = explainer.explain_instance(encoder.encode(customer)[0], proba, num_features=top, num_samples=samples)
    return pd.Series(dict(exp.as_list(label=1)))


def lime_features(explainer, predict: Predict, encoder: RawEncoder, customer: pd.DataFrame, top: int = 5) -> list[str]:
    """Names of the inputs LIME ranks highest for one customer (to compare with SHAP)."""

    def proba(matrix: np.ndarray) -> np.ndarray:
        p = np.asarray(predict(encoder.decode(matrix)), dtype=float)
        return np.column_stack([1 - p, p])

    exp = explainer.explain_instance(encoder.encode(customer)[0], proba, num_features=top, num_samples=2000)
    return [encoder.columns[i] for i, _ in exp.as_map()[1]]


def top_agreement(shap_features: list[str], lime_feature_names: list[str]) -> float:
    """Share of LIME's top inputs that are also among SHAP's top inputs for the same customer."""
    if not lime_feature_names:
        return 0.0
    return len(set(shap_features) & set(lime_feature_names)) / len(lime_feature_names)
