"""Train the churn model: compare models and feature sets by CV, tune, calibrate, gate and register."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.training.business import best_row, profit_curve, selection_gap
from src.training.data import PROCESSED_PATH, load_params
from src.training.features import FeatureEngineer
from src.training.model import ChurnProbaModel
from src.training.registry import CHALLENGER, MODEL_NAME, champion_pr_auc, passes_gate, provenance

EXPERIMENT = "churn-prediction"
REPORT_PATH = Path("reports/drift.html")
METRICS_PATH = Path("reports/metrics.json")
XGB_DEFAULTS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}


def make_estimator(name: str, seed: int, scale_pos_weight: float = 1.0, **params):
    """Classifier by name: dummy (prior), logreg, rf or xgb."""
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    if name == "xgb":
        return XGBClassifier(
            **{**XGB_DEFAULTS, **params},
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model: {name}")


def build_pipeline(estimator, add_features: bool = True, drop_sensitive: bool = False) -> Pipeline:
    """Feature engineering, impute/scale numerics, one-hot categoricals, then the classifier."""
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include=object),
            ),
        ]
    )
    return Pipeline(
        [("fe", FeatureEngineer(add_features, drop_sensitive)), ("prep", prep), ("clf", estimator)]
    )


def config_name(cfg: dict) -> str:
    """Readable name for one (model, features, sensitive) configuration."""
    sens = "nosens" if cfg["drop_sensitive"] else "sens"
    return f"{cfg['model']}{'-tuned' if cfg.get('params') else ''}-fe{int(cfg['add_features'])}-{sens}"


def cv_config(cfg: dict, X: pd.DataFrame, y: pd.Series, params: dict, spw: float) -> dict:
    """Out-of-fold PR-AUC and group selection gaps for one configuration, logged as a nested run."""
    cv = StratifiedKFold(params["cv_folds"], shuffle=True, random_state=params["seed"])
    est = make_estimator(cfg["model"], params["seed"], spw, **cfg.get("params", {}))
    pipe = build_pipeline(est, cfg["add_features"], cfg["drop_sensitive"])
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    res = {
        **cfg,
        "cv_pr_auc": average_precision_score(y, oof),
        "dpd_gender": selection_gap(oof, X["gender"]),
        "dpd_senior": selection_gap(oof, X["SeniorCitizen"]),
    }
    with mlflow.start_run(run_name=config_name(cfg), nested=True):
        mlflow.log_params({**{k: v for k, v in cfg.items() if k != "params"}, "add_features": int(cfg["add_features"]),
                           "drop_sensitive": int(cfg["drop_sensitive"])})
        mlflow.log_metrics({k: v for k, v in res.items() if k in ("cv_pr_auc", "dpd_gender", "dpd_senior")})
    return res


def compare_configs(X: pd.DataFrame, y: pd.Series, params: dict, spw: float) -> list[dict]:
    """CV every model x feature set x sensitive-column setting at default hyper-parameters."""
    return [
        cv_config({"model": m, "add_features": f, "drop_sensitive": d, "params": {}}, X, y, params, spw)
        for m in params["models"]
        for f in (False, True)
        for d in (False, True)
    ]


def select_config(results: list[dict], tolerance: float, models: list[str] | None = None) -> dict:
    """Best CV PR-AUC; prefer dropping sensitive columns when it costs at most `tolerance`."""
    pool = [r for r in results if models is None or r["model"] in models]
    best = max(pool, key=lambda r: r["cv_pr_auc"])
    if not best["drop_sensitive"]:
        twin = next(
            (r for r in pool if r["model"] == best["model"] and r["add_features"] == best["add_features"]
             and r["params"] == best["params"] and r["drop_sensitive"]),
            None,
        )
        if twin is not None and twin["cv_pr_auc"] >= best["cv_pr_auc"] - tolerance:
            return twin
    return best


def tune(X: pd.DataFrame, y: pd.Series, cfg: dict, params: dict, spw: float) -> dict:
    """Optuna search over XGBoost maximising CV PR-AUC; every trial is a nested MLflow run.

    Tuning uses mean-fold PR-AUC; the tuned result is re-scored with `cv_config` (pooled
    out-of-fold PR-AUC) so it is comparable with the other candidates.
    """
    cv = StratifiedKFold(params["cv_folds"], shuffle=True, random_state=params["seed"])

    def objective(trial: optuna.Trial) -> float:
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        pipe = build_pipeline(
            make_estimator("xgb", params["seed"], spw, **xgb_params), cfg["add_features"], cfg["drop_sensitive"]
        )
        score = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision").mean()
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(xgb_params)
            mlflow.log_metric("cv_pr_auc", score)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=params["seed"]))
    study.optimize(objective, n_trials=params["n_trials"])
    mlflow.log_metric("best_cv_pr_auc", study.best_value)
    return study.best_params


def evaluate(pipe, X: pd.DataFrame, y: pd.Series, econ: dict) -> tuple[dict, pd.DataFrame]:
    """Test-set metrics plus the contact-budget curve.

    precision/recall/f1 are measured at the profit-optimal contact share `best_k`,
    not at a fixed 0.5 probability cut-off.
    """
    proba = pipe.predict_proba(X)[:, 1]
    curve = profit_curve(y.to_numpy(), proba, X["MonthlyCharges"].to_numpy(), econ)
    best = best_row(curve)
    p, r = float(best["precision"]), float(best["recall"])
    metrics = {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "best_k": float(best["k"]),
        "best_profit": float(best["profit"]),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "dpd_gender": selection_gap(proba, X["gender"]),
        "dpd_senior": selection_gap(proba, X["SeniorCitizen"]),
    }
    return metrics, curve


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Evidently report, train vs test. A sanity check of the random split, not drift detection."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))


def log_figures(curve: pd.DataFrame, raw_proba: np.ndarray, cal_proba: np.ndarray, y: pd.Series) -> None:
    """Profit-by-k and calibration (before/after) plots as MLflow artifacts."""
    fig, ax = plt.subplots()
    ax.plot(curve["k"] * 100, curve["profit"], marker="o")
    ax.set(xlabel="Contacted customers (% of top scores)", ylabel="Realised profit", title="Profit by contact share")
    mlflow.log_figure(fig, "profit_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, proba in (("uncalibrated", raw_proba), ("isotonic", cal_proba)):
        frac, mean = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set(xlabel="Predicted probability", ylabel="Observed churn rate", title="Calibration")
    ax.legend()
    mlflow.log_figure(fig, "calibration.png")
    plt.close(fig)


def run_training(df: pd.DataFrame, params: dict) -> dict:
    """Full training run; returns the test metrics plus gate result."""
    train_p, gate_p, econ = params["train"], params["gate"], params["economics"]
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=train_p["test_size"], stratify=y, random_state=train_p["seed"]
    )
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    seed = train_p["seed"]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tags(provenance())
        mlflow.log_params({f"train.{k}": v for k, v in train_p.items()})
        mlflow.log_params({f"economics.{k}": v for k, v in econ.items() if k != "k_grid"})
        mlflow.log_param("scale_pos_weight", round(spw, 3))

        # Model, feature and sensitive-column choices are made on CV only; the test set is used once below.
        results = compare_configs(X_tr, y_tr, train_p, spw)
        tol = train_p["sensitive_tolerance"]
        if "xgb" in train_p["models"]:  # tune XGBoost as a full candidate, not only if it wins at defaults
            xgb_cfg = select_config(results, tol, models=["xgb"])
            tuned = tune(X_tr, y_tr, xgb_cfg, train_p, spw)
            tuned_cfg = {k: xgb_cfg[k] for k in ("model", "add_features", "drop_sensitive")}
            results.append(cv_config({**tuned_cfg, "params": tuned}, X_tr, y_tr, train_p, spw))
        best = select_config(results, tol)
        best_params = best["params"]
        mlflow.log_params({f"selected.{k}": best[k] for k in ("model", "add_features", "drop_sensitive")})
        mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})
        mlflow.log_metric("selected_cv_pr_auc", best["cv_pr_auc"])

        def fresh(scale_pos_weight: float = spw):
            est = make_estimator(best["model"], seed, scale_pos_weight, **best_params)
            return build_pipeline(est, best["add_features"], best["drop_sensitive"])

        raw = fresh().fit(X_tr, y_tr)
        cv = StratifiedKFold(train_p["cv_folds"], shuffle=True, random_state=seed)
        calibrated = CalibratedClassifierCV(fresh(), method="isotonic", cv=cv).fit(X_tr, y_tr)
        metrics, curve = evaluate(calibrated, X_te, y_te, econ)
        raw_proba = raw.predict_proba(X_te)[:, 1]
        metrics["brier_uncalibrated"] = brier_score_loss(y_te, raw_proba)
        mlflow.log_metrics(metrics)
        mlflow.log_text(curve.to_csv(index=False), "profit_curve.csv")
        log_figures(curve, raw_proba, calibrated.predict_proba(X_te)[:, 1], y_te)

        if best["model"] == "xgb":  # side experiment from Q4: does dropping scale_pos_weight help?
            with mlflow.start_run(run_name="xgb-no-scale-pos-weight", nested=True):
                no_spw = fresh(1.0)
                score = cross_val_score(no_spw, X_tr, y_tr, cv=cv, scoring="average_precision").mean()
                mlflow.log_metric("cv_pr_auc", score)

        # The baseline only makes sense when a more complex model won; a simple winner is not compared with itself.
        baseline_pr_auc = None
        if best["model"] not in ("dummy", "logreg"):
            baseline = select_config(results, tol, models=["logreg"])
            base_pipe = build_pipeline(
                make_estimator("logreg", seed), baseline["add_features"], baseline["drop_sensitive"]
            ).fit(X_tr, y_tr)
            baseline_pr_auc = float(average_precision_score(y_te, base_pipe.predict_proba(X_te)[:, 1]))
            mlflow.log_metric("baseline_pr_auc", baseline_pr_auc)

        drift_report(X_tr, X_te)
        mlflow.log_artifact(str(REPORT_PATH))

        champion = champion_pr_auc(MlflowClient())
        passed, reasons = passes_gate(metrics["pr_auc"], baseline_pr_auc, champion, gate_p)
        model = ChurnProbaModel(calibrated)
        info = mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            signature=infer_signature(X_te.head(5), model.predict(None, X_te.head(5))),
            input_example=X_te.head(3),
            registered_model_name=MODEL_NAME if passed else None,
        )
        if passed:
            MlflowClient().set_registered_model_alias(MODEL_NAME, CHALLENGER, info.registered_model_version)
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")
        mlflow.set_tag("gate_failures", "; ".join(reasons))

    return {
        **metrics,
        "baseline_pr_auc": baseline_pr_auc,
        "passed": passed,
        "reasons": reasons,
        "version": str(info.registered_model_version) if passed else None,
        "run_id": run.info.run_id,
    }


def main() -> None:
    """DVC `train` stage."""
    outcome = run_training(pd.read_csv(PROCESSED_PATH), load_params())
    metrics = {k: round(float(v), 4) for k, v in outcome.items() if isinstance(v, (float, np.floating))}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    if not outcome["passed"]:
        print("Quality gate failed: " + "; ".join(outcome["reasons"]), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"Registered {MODEL_NAME} v{outcome['version']} as {CHALLENGER} (run {outcome['run_id']}). "
        "It is not served until approved: use the Mô hình tab of the web UI, or python -m src.training.promote."
    )


if __name__ == "__main__":
    main()
