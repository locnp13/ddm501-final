"""Train the churn model: Optuna + CV, MLflow tracking, Evidently report, quality gate."""
import json
import sys
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.training.data import PROCESSED_PATH, load_params
from src.training.model import ChurnProbaModel

MODEL_NAME = "churn-model"
EXPERIMENT = "churn-prediction"
REPORT_PATH = Path("reports/drift.html")
METRICS_PATH = Path("reports/metrics.json")
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def passes_gate(pr_auc: float, min_pr_auc: float) -> bool:
    """Quality gate: the model may go to Production only at or above the threshold."""
    return pr_auc >= min_pr_auc


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float, seed: int, **xgb_params) -> Pipeline:
    """Preprocessing (impute + one-hot) followed by XGBoost."""
    categorical = [c for c in X.columns if c not in NUMERIC and X[c].dtype == "object"]
    passthrough = [c for c in X.columns if c not in NUMERIC and c not in categorical]
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("pass", "passthrough", passthrough),
        ]
    )
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **xgb_params,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def tune(X: pd.DataFrame, y: pd.Series, params: dict, scale_pos_weight: float) -> dict:
    """Optuna search maximising CV PR-AUC; every trial is a nested MLflow run."""
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
        pipe = build_pipeline(X, scale_pos_weight, params["seed"], **xgb_params)
        score = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision").mean()
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(xgb_params)
            mlflow.log_metric("cv_pr_auc", score)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=params["seed"])
    )
    study.optimize(objective, n_trials=params["n_trials"])
    mlflow.log_metric("best_cv_pr_auc", study.best_value)
    return study.best_params


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Test-set metrics; precision/recall/F1 use a 0.5 probability cut-off."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Evidently data-drift report, train (reference) vs test (current)."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))


def main() -> None:
    """DVC `train` stage."""
    params = load_params()
    train_p, gate_p = params["train"], params["gate"]

    df = pd.read_csv(PROCESSED_PATH)
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=train_p["test_size"], stratify=y, random_state=train_p["seed"]
    )
    scale_pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params({f"train.{k}": v for k, v in train_p.items()})
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
        best = tune(X_tr, y_tr, train_p, scale_pos_weight)
        mlflow.log_params({f"best.{k}": v for k, v in best.items()})

        pipe = build_pipeline(X_tr, scale_pos_weight, train_p["seed"], **best).fit(X_tr, y_tr)
        metrics = evaluate(pipe, X_te, y_te)
        mlflow.log_metrics(metrics)

        drift_report(X_tr, X_te)
        mlflow.log_artifact(str(REPORT_PATH))

        passed = passes_gate(metrics["pr_auc"], gate_p["min_pr_auc"])
        model = ChurnProbaModel(pipe)
        signature = infer_signature(X_te.head(5), model.predict(None, X_te.head(5)))
        info = mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            signature=signature,
            input_example=X_te.head(3),
            registered_model_name=MODEL_NAME if passed else None,
        )
        if passed:
            client = MlflowClient()
            client.transition_model_version_stage(
                MODEL_NAME, info.registered_model_version, "Production", archive_existing_versions=True
            )
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
    print(json.dumps(metrics, indent=2))
    if not passed:
        print(
            f"Quality gate failed: PR-AUC {metrics['pr_auc']:.3f} < {gate_p['min_pr_auc']}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Registered {MODEL_NAME} v{info.registered_model_version} as Production (run {run.info.run_id})")


if __name__ == "__main__":
    main()
