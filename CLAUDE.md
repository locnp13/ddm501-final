# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This directory is currently empty apart from `DDM501_Final_Project.pdf` (the course brief). There is no code, git repo, build system, or tests yet. Update this file with real commands and architecture once the stack is chosen and code exists.

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
