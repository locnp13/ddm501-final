"""Responsible-AI report for a registered churn model: explanations (SHAP, LIME) and fairness.

Run it in the trainer container after a model is approved (default: the champion)::

    docker compose run --rm trainer python -m src.responsible.report
    docker compose run --rm trainer python -m src.responsible.report --model-uri models:/churn-model/3 --log-to-run

It rebuilds the training/test split of the training stage (same seed and size from params.yaml), scores the
test customers with the registered model, and writes to reports/responsible_ai/:

- results.json: every number below;
- summary.md: the same numbers as a readable report (Vietnamese);
- shap_importance.png, local_explanations.png, fairness_senior.png.

With --log-to-run the folder and the headline metrics are also logged to the model's MLflow run.
Discussion of what the numbers mean, privacy and ethics: docs/responsible-ai.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.responsible.explain import (
    RawEncoder,
    lime_explain,
    lime_explainer,
    lime_features,
    shap_explain,
    top_agreement,
)
from src.responsible.fairness import (
    apply_thresholds,
    disparities,
    equal_opportunity_thresholds,
    group_metrics,
    mitigation_study,
)
from src.training.business import top_k_mask
from src.training.data import PROCESSED_PATH, load_params
from src.training.features import SENSITIVE_COLS

OUT_DIR = Path("reports/responsible_ai")
DEFAULT_URI = "models:/churn-model@champion"
CATEGORY_BREAKDOWN = ("Contract", "InternetService", "PaymentMethod", "TechSupport")


def split(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """The training stage's train/test split (stratified on Churn, same seed and size)."""
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    return train_test_split(X, y, test_size=params["test_size"], stratify=y, random_state=params["seed"])


def proxy_strength(X_tr: pd.DataFrame, X_te: pd.DataFrame, column: str) -> float:
    """ROC-AUC of predicting a sensitive column from the other inputs: how much the model could infer it anyway."""
    others = [c for c in X_tr.columns if c not in SENSITIVE_COLS]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(pd.get_dummies(X_tr[others], dtype=float).fillna(0), X_tr[column])
    test = pd.get_dummies(X_te[others], dtype=float).reindex(columns=clf.feature_names_in_, fill_value=0).fillna(0)
    return float(roc_auc_score(X_te[column], clf.predict_proba(test)[:, 1]))


def budget_of(run_id: str | None, params: dict) -> float:
    """Contact share used by the model's training run (profit-optimal k), or the middle of the k grid."""
    if run_id:
        try:
            k = MlflowClient().get_run(run_id).data.metrics.get("best_k")
            if k:
                return float(k)
        except Exception:  # registry not reachable: fall back
            pass
    grid = params["economics"]["k_grid"]
    return float(grid[len(grid) // 2])


def analyse(
    predict, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_te: pd.Series, budget: float, econ: dict,
    n_explain: int = 200, n_background: int = 50, n_lime: int = 20, repeats: int = 50, seed: int = 42,
) -> dict[str, Any]:
    """All responsible-AI numbers for one model; `predict` maps a raw frame to churn probabilities."""
    rng = np.random.default_rng(seed)
    encoder = RawEncoder(pd.concat([X_tr, X_te]))
    score = np.asarray(predict(X_te), dtype=float)

    # Explanations on a random sample of test customers, against a sample of training customers.
    background = X_tr.sample(min(n_background, len(X_tr)), random_state=seed)
    explain_idx = rng.choice(len(X_te), size=min(n_explain, len(X_te)), replace=False)
    explained = X_te.iloc[explain_idx]
    shap_res = shap_explain(predict, encoder, background, explained, seed=seed)
    explained_score = score[explain_idx]
    additivity = float(np.abs(shap_res.base_value + shap_res.values.sum(axis=1) - explained_score).max())

    lime = lime_explainer(encoder, X_tr.sample(min(1000, len(X_tr)), random_state=seed), seed=seed)
    agreement = []
    for i in range(min(n_lime, len(explained))):
        shap_top = list(pd.Series(np.abs(shap_res.values[i]), index=encoder.columns).nlargest(5).index)
        agreement.append(top_agreement(shap_top, lime_features(lime, predict, encoder, explained.iloc[[i]])))

    examples = {"high_risk": int(np.argmax(explained_score)), "low_risk": int(np.argmin(explained_score))}
    local = {}
    for name, i in examples.items():
        local[name] = {
            "probability": float(explained_score[i]),
            "customer": {k: (None if pd.isna(v) else v) for k, v in explained.iloc[i].to_dict().items()},
            "shap": shap_res.local(i).round(4).to_dict(),
            "lime": lime_explain(lime, predict, encoder, explained.iloc[[i]]).round(4).to_dict(),
        }

    # Fairness of the contact decision (top `budget` share of the test customers).
    selected = top_k_mask(score, budget)
    fairness = {}
    for column in SENSITIVE_COLS:
        metrics = group_metrics(y_te.to_numpy(), selected, score, X_te[column])
        fairness[column] = {"groups": metrics.round(4).to_dict(orient="index"),
                            "gaps": {k: round(v, 4) for k, v in disparities(metrics).items()},
                            "proxy_auc": round(proxy_strength(X_tr, X_te, column), 4)}

    study = mitigation_study(score, y_te.to_numpy(), X_te["SeniorCitizen"], X_te["MonthlyCharges"].to_numpy(),
                             budget, econ, repeats=repeats, seed=seed)
    summary = study.groupby("decision")[["tpr_gap", "selection_gap", "contact_share", "profit"]].agg(["mean", "std"])
    thresholds = equal_opportunity_thresholds(score, y_te.to_numpy(), X_te["SeniorCitizen"], budget)
    mitigated = group_metrics(y_te.to_numpy(), apply_thresholds(score, X_te["SeniorCitizen"], thresholds), score,
                              X_te["SeniorCitizen"])

    return {
        "budget": budget,
        "test_customers": int(len(X_te)),
        "explained_customers": int(len(explained)),
        "background_customers": int(len(background)),
        "shap": {
            "base_value": round(shap_res.base_value, 4),
            "max_additivity_error": round(additivity, 6),
            "importance": shap_res.importance().round(4).to_dict(),
            "by_category": {c: shap_res.by_category(c).round(4).to_dict() for c in CATEGORY_BREAKDOWN},
        },
        "lime_shap_top5_agreement": round(float(np.mean(agreement)), 3),
        "local": local,
        "fairness": fairness,
        "mitigation": {
            "attribute": "SeniorCitizen",
            "repeats": repeats,
            "summary": {f"{d}.{m}.{s}": round(float(v), 4) for (m, s), row in summary.T.iterrows()
                        for d, v in row.items()},
            "thresholds_full_test": {str(k): round(float(v), 4) for k, v in thresholds.items()},
            "groups_full_test": mitigated.round(4).to_dict(orient="index"),
        },
    }


def plot(results: dict[str, Any], out: Path) -> None:
    """The three figures of the report."""
    imp = pd.Series(results["shap"]["importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set(xlabel="Mean |SHAP| (change in churn probability)", title="Which inputs drive the prediction (SHAP)")
    fig.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, name in enumerate(("high_risk", "low_risk")):
        item = results["local"][name]
        for col, method in enumerate(("shap", "lime")):
            series = pd.Series(item[method]).iloc[::-1]
            colors = ["#c0392b" if v > 0 else "#2e86c1" for v in series.values]
            axes[row, col].barh(series.index, series.values, color=colors)
            axes[row, col].axvline(0, color="black", linewidth=0.8)
            title = f"{method.upper()}: {name.replace('_', ' ')} customer, p={item['probability']:.2f}"
            axes[row, col].set_title(title)
            axes[row, col].tick_params(axis="y", labelsize=8)
    fig.suptitle("Why this customer? Red pushes towards churn, blue towards staying")
    fig.tight_layout()
    fig.savefig(out / "local_explanations.png", dpi=120)
    plt.close(fig)

    groups = pd.DataFrame(results["fairness"]["SeniorCitizen"]["groups"]).T
    mitigated = pd.DataFrame(results["mitigation"]["groups_full_test"]).T
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(groups))
    for offset, (label, frame) in zip((-0.2, 0.2), (("top-k (current)", groups), ("equal opportunity", mitigated))):
        axes[0].bar(x + offset, frame["tpr"], width=0.4, label=label)
    axes[0].set_xticks(x, [f"SeniorCitizen={g}" for g in groups.index])
    axes[0].set(ylabel="Churners contacted (TPR)", ylim=(0, 1), title="Who among churners gets an offer")
    axes[0].legend()
    for offset, (label, frame) in zip((-0.2, 0.2), (("top-k (current)", groups), ("equal opportunity", mitigated))):
        axes[1].bar(x + offset, frame["selection_rate"], width=0.4, label=label)
    axes[1].bar(x, groups["churn_rate"], width=0.05, color="black", label="actual churn rate")
    axes[1].set_xticks(x, [f"SeniorCitizen={g}" for g in groups.index])
    axes[1].set(ylabel="Share contacted", ylim=(0, 1), title="Selection rate per group")
    axes[1].legend()
    fig.suptitle("Contact decision by SeniorCitizen (group thresholds fitted on the whole test set: illustration; "
                 "see summary.md for held-out results)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fairness_senior.png", dpi=120)
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_summary(results: dict[str, Any], out: Path, model_uri: str, version: str | None) -> None:
    """Readable summary of results.json, regenerated on every run."""
    imp = results["shap"]["importance"]
    senior = results["fairness"]["SeniorCitizen"]
    gender = results["fairness"]["gender"]
    m = results["mitigation"]["summary"]
    lines = [
        "# Báo cáo Responsible AI (tự sinh)",
        "",
        f"Tạo bởi `python -m src.responsible.report` cho model `{model_uri}`"
        + (f" (phiên bản {version})" if version else "") + ". Giải thích ý nghĩa: `docs/responsible-ai.md`.",
        "",
        f"Tập test: {results['test_customers']} khách. Quyết định đánh giá: liên hệ {_pct(results['budget'])} "
        "khách có điểm cao nhất (k tối ưu lợi nhuận của lần huấn luyện).",
        "",
        "## 1. Giải thích mô hình",
        "",
        f"SHAP (permutation, {results['explained_customers']} khách, nền {results['background_customers']} khách "
        f"huấn luyện; xác suất trung bình nền {results['shap']['base_value']:.3f}; sai số cộng tính lớn nhất "
        f"{results['shap']['max_additivity_error']:.1e}).",
        "",
        "| Đầu vào | Trung bình \\|SHAP\\| |",
        "|---|---|",
        *[f"| `{k}` | {v:.4f} |" for k, v in list(imp.items())[:10]],
        "",
        "Ảnh hưởng trung bình theo giá trị (dương: đẩy về phía rời bỏ):",
        "",
    ]
    for column, values in results["shap"]["by_category"].items():
        lines.append(f"- `{column}`: " + ", ".join(f"{k} {v:+.3f}" for k, v in values.items()))
    lines += [
        "",
        f"LIME so với SHAP: trung bình {_pct(results['lime_shap_top5_agreement'])} trong 5 đầu vào LIME xếp cao nhất "
        "cũng nằm trong top 5 của SHAP (cùng khách).",
        "",
        "## 2. Công bằng của quyết định liên hệ",
        "",
        "| Nhóm | Số khách | Tỷ lệ churn thật | Xác suất TB | Tỷ lệ được liên hệ | TPR | FPR | Precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for column, block in (("SeniorCitizen", senior), ("gender", gender)):
        for g, row in block["groups"].items():
            lines.append(
                f"| {column}={g} | {row['n']:.0f} | {_pct(row['churn_rate'])} | {_pct(row['mean_score'])} | "
                f"{_pct(row['selection_rate'])} | {_pct(row['tpr'])} | {_pct(row['fpr'])} | {_pct(row['precision'])} |"
            )
    lines += [
        "",
        f"- `SeniorCitizen`: chênh tỷ lệ được liên hệ {_pct(senior['gaps']['demographic_parity_diff'])}, "
        f"chênh TPR {_pct(senior['gaps']['equal_opportunity_diff'])}, tỷ số disparate impact "
        f"{senior['gaps']['disparate_impact_ratio']:.2f}. Đoán `SeniorCitizen` từ các cột còn lại: ROC-AUC "
        f"{senior['proxy_auc']:.2f}.",
        f"- `gender`: chênh tỷ lệ được liên hệ {_pct(gender['gaps']['demographic_parity_diff'])}, chênh TPR "
        f"{_pct(gender['gaps']['equal_opportunity_diff'])}. Đoán `gender` từ các cột còn lại: ROC-AUC "
        f"{gender['proxy_auc']:.2f}.",
        "",
        "## 3. Giảm thiên lệch: ngưỡng theo nhóm (equal opportunity) cho `SeniorCitizen`",
        "",
        f"Ngưỡng học trên một nửa tập test, đánh giá trên nửa còn lại, lặp {results['mitigation']['repeats']} lần "
        "(trung bình ± độ lệch chuẩn), cùng ngân sách liên hệ.",
        "",
        "| Quyết định | Chênh TPR | Chênh tỷ lệ được liên hệ | Tỷ lệ liên hệ | Lợi nhuận (nửa tập test) |",
        "|---|---|---|---|---|",
    ]
    for d, label in (("baseline", "top-k hiện tại"), ("equal_opportunity", "ngưỡng theo nhóm")):
        lines.append(
            f"| {label} | {_pct(m[f'{d}.tpr_gap.mean'])} ± {_pct(m[f'{d}.tpr_gap.std'])} | "
            f"{_pct(m[f'{d}.selection_gap.mean'])} | {_pct(m[f'{d}.contact_share.mean'])} | "
            f"{m[f'{d}.profit.mean']:,.0f} ± {m[f'{d}.profit.std']:,.0f} |"
        )
    lines += ["", "Hình: `shap_importance.png`, `local_explanations.png`, `fairness_senior.png`.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Explanations (SHAP, LIME) and fairness report for a churn model.")
    parser.add_argument("--model-uri", default=DEFAULT_URI)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--data", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--explain", type=int, default=200, help="test customers explained with SHAP")
    parser.add_argument("--background", type=int, default=50, help="training customers used as SHAP background")
    parser.add_argument("--lime", type=int, default=20, help="customers compared between LIME and SHAP")
    parser.add_argument("--repeats", type=int, default=50, help="random half-splits for the mitigation study")
    parser.add_argument("--log-to-run", action="store_true", help="log the report to the model's MLflow run")
    args = parser.parse_args(argv)

    params = load_params()
    model = mlflow.pyfunc.load_model(args.model_uri)
    run_id = getattr(model.metadata, "run_id", None)
    X_tr, X_te, _, y_te = split(pd.read_csv(args.data), params["train"])
    budget = budget_of(run_id, params)
    results = analyse(
        model.predict, X_tr, X_te, y_te, budget, params["economics"], n_explain=args.explain,
        n_background=args.background, n_lime=args.lime, repeats=args.repeats, seed=params["train"]["seed"],
    )
    version = None
    try:
        version = next((str(v.version) for v in MlflowClient().search_model_versions(f"run_id='{run_id}'")), None)
    except Exception:
        pass
    results = {"model_uri": args.model_uri, "model_version": version, "run_id": run_id, **results}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    plot(results, args.out)
    write_summary(results, args.out, args.model_uri, version)
    if args.log_to_run and run_id:
        senior = results["fairness"]["SeniorCitizen"]["gaps"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(args.out), "responsible_ai")
            mlflow.log_metrics({
                "rai_senior_selection_gap": senior["demographic_parity_diff"],
                "rai_senior_tpr_gap": senior["equal_opportunity_diff"],
                "rai_lime_shap_agreement": results["lime_shap_top5_agreement"],
            })
    print(f"Wrote {args.out}/summary.md, results.json and 3 figures for {args.model_uri} (v{version}).")
    return results


if __name__ == "__main__":
    main()
