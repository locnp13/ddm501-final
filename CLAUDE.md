# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Git repo (origin `github.com/locnp13/ddm501-final`, branch `main`). Training pipeline, API skeleton, Compose stack and CI exist; see `README.md` for status and `docs/open-questions.md` for undecided points. Still missing: Grafana dashboards, SHAP/LIME, fairness analysis, batch endpoint, CI deploy step.

## Commands

Everything runs in Docker; no Python install on the host is needed.

- Start stack: `docker compose up -d --build` (MLflow on host port 5001, MinIO 9000/9001, API 8000, web UI 8080, Prometheus 9090, Grafana 3000)
- Train (DVC: `ingest` then `train`; compares 4 models x feature sets x sensitive-column setting by CV, tunes XGBoost, calibrates, and if the 3-part gate passes registers the model with alias `challenger`): `GIT_COMMIT=$(git rev-parse HEAD) GIT_DIRTY=$(git status --porcelain | wc -l) docker compose run --rm trainer dvc repro` (the env vars tag the MLflow run for reproducibility)
- Approve a model (human step): `docker compose run --rm trainer python -m src.training.promote [VERSION]` sets alias `champion`; the API serves `models:/churn-model@champion`
- Reload model in API (loaded only at startup): `docker compose restart api`
- Tests: `docker compose run --rm trainer pytest -q`; lint: `ruff check src tests` (config in `ruff.toml`)
- Data remote is the `dvc` bucket on MinIO: `docker compose run --rm trainer dvc push|pull`

## Architecture

`src/training/data.py` (download, Pandera schema, clean) -> `features.py` (in-pipeline feature engineering), `business.py` (profit/Recall@k), `registry.py` (gate, aliases, provenance), `train.py` (CV comparison, Optuna, isotonic calibration, MLflow logging, Evidently report) -> `model.py` (pyfunc wrapper returning P(churn)). `src/serving/app.py` loads `models:/churn-model@champion` and exposes `/v1/predict` (input validated by `src/serving/schemas.py`; keep it in sync with the Pandera `SCHEMA`, a test checks this), `/v1/model`, `/health`, `/metrics`. `frontend/` is an nginx container serving a static UI (plain HTML/CSS/JS) that calls the API through its `/api` proxy; `src/serving/model_info.py` builds the model description the UI shows from the MLflow registry. MLflow metadata is in Postgres and artifacts in MinIO (`mlflow` bucket). Pin xgboost/scikit-learn identically in `requirements.txt` (API) and `requirements-train.txt` (trainer) or the model will not load.

## What the project is

Capstone for DDM501 "AI in Production" (team of 3-4, 40% of the course grade, presented in Session 10). The goal is an end-to-end ML system, from problem definition to production deployment.

**Chosen topic: Topic 3, Customer Churn Prediction (E-Commerce & Retail).** The PDF only gives a short brief: identify customers likely to churn and enable proactive retention. Suggested approaches are classification models, survival analysis and deep learning. Key challenges are imbalanced classes, feature engineering and model explainability. Suggested datasets are Telco Customer Churn and the E-Commerce Churn Dataset. The problem statement, metrics and architecture are for the team to define.

## Required deliverables (from the PDF; these define the target architecture)

The repo will be graded as a GitHub repo, so structure it to make each item easy to find:

- **ML pipeline**: data ingestion/validation/versioning, feature engineering, training with multiple experiments, hyperparameter tuning and cross-validation, and MLflow (or equivalent) tracking of metrics, params and artifacts.
- **Serving**: a REST API with error handling, versioning and Swagger/OpenAPI docs. It needs a Dockerfile (multi-stage, security best practices) and a docker-compose file with all services and health checks.
- **Monitoring**: Prometheus metrics, including custom ML metrics as well as system metrics, plus Grafana dashboards and alerting rules with meaningful thresholds.
- **Testing/CI**: unit, integration (API), data-quality and model-validation tests, with more than 80% coverage. A GitHub Actions pipeline should run lint, test, build and deploy.
- **Responsible AI**: fairness/bias analysis with mitigation, explainability with more than one method (SHAP and LIME), a data-privacy discussion and an ethics discussion.
- **Docs**: a README with badges, examples and troubleshooting; an OpenAPI spec with examples; a deployment/operation guide; an architecture diagram and data-flow diagram; and a tech-stack justification with trade-offs. Code should have type hints, docstrings and a consistent style.

Rubric weights: problem definition 10%, design 15%, implementation 40% (ML pipeline 15, deployment 15, monitoring 10), testing/CI 15%, responsible AI 10%, docs 10%. Individual grades can be adjusted by ±20% based on contribution, so keep commits attributable.
