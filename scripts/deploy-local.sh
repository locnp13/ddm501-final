#!/usr/bin/env bash
# Build api and frontend for the local minikube cluster and roll them out.
# Usage: scripts/deploy-local.sh [TAG]   (TAG defaults to the short commit; CI passes the commit SHA)
# Needs minikube profile "churn" (started if stopped), docker and kubectl. MLflow/MinIO/Postgres
# (docker compose) must already be running; the API pods read the champion model from MLflow.
set -euo pipefail

# A launchd runner has a minimal PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.docker/bin:$PATH"

PROFILE=churn
NS=churn
TAG="${1:-$(git rev-parse --short=12 HEAD)}"
TAG="${TAG:0:12}"
cd "$(dirname "$0")/.."

echo "== Pre-flight checks on $(hostname -s)"
docker info >/dev/null 2>&1 || { echo "ERROR: Docker is not running (start Docker Desktop)"; exit 1; }
curl -fsS --max-time 5 http://localhost:5001/health >/dev/null \
  || { echo "ERROR: MLflow is not reachable on localhost:5001 (docker compose up -d mlflow minio postgres)"; exit 1; }
echo "docker: ok, mlflow: ok"

if ! minikube -p "$PROFILE" status >/dev/null 2>&1; then
  echo "Cluster '$PROFILE' is not running; starting it"
  minikube start -p "$PROFILE" --driver=docker --cpus=4 --memory=4096
fi
kubectl --context "$PROFILE" -n "$NS" get secret churn-secrets >/dev/null  # created once by hand, see k8s/README.md

echo "== Building images tagged $TAG inside the cluster's Docker"
eval "$(minikube -p "$PROFILE" docker-env)"
docker build -q -t "churn-api:$TAG" -f deploy/Dockerfile.api .
docker build -q -t "churn-frontend:$TAG" -f frontend/Dockerfile frontend

echo "== Applying manifests with the new tag"
kubectl --context "$PROFILE" kustomize k8s/base \
  | sed -e "s#churn-api:dev#churn-api:$TAG#" -e "s#churn-frontend:dev#churn-frontend:$TAG#" \
  | kubectl --context "$PROFILE" apply -f -

echo "== Waiting for the rollout"
kubectl --context "$PROFILE" -n "$NS" rollout status deploy/api --timeout=300s
kubectl --context "$PROFILE" -n "$NS" rollout status deploy/frontend --timeout=120s

echo "== Smoke test: the API answers and has a model"
for _ in $(seq 1 30); do
  health="$(kubectl --context "$PROFILE" -n "$NS" exec deploy/frontend -- wget -qO- http://api:8000/health || true)"
  [[ "$health" == *'"model_loaded":true'* ]] && break
  sleep 2
done
echo "$health"
[[ "$health" == *'"model_loaded":true'* ]] || { echo "API is up but has no model (is there a champion in MLflow?)"; exit 1; }

echo "== Removing old images (keeping the 3 newest besides $TAG)"
docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^churn-(api|frontend):' | grep -v -E ":($TAG|dev)$" \
  | tail -n +7 | xargs -r docker rmi >/dev/null 2>&1 || true
echo "Deployed $TAG"
