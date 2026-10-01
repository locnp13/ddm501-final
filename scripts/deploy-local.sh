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
# churn-secrets is created by hand from .env (see k8s/README.md); it must hold both keys.
for key in ADMIN_KEY GRAFANA_ADMIN_PASSWORD; do
  value="$(kubectl --context "$PROFILE" -n "$NS" get secret churn-secrets -o "jsonpath={.data.$key}" 2>/dev/null || true)"
  [[ -n "$value" ]] || { echo "ERROR: secret churn-secrets has no $key (kubectl -n $NS create secret generic churn-secrets --from-env-file=.env)"; exit 1; }
done

echo "== Building images tagged $TAG inside the cluster's Docker"
eval "$(minikube -p "$PROFILE" docker-env)"
docker build -q -t "churn-api:$TAG" -f deploy/Dockerfile.api .
docker build -q -t "churn-frontend:$TAG" -f frontend/Dockerfile frontend

echo "== Applying manifests with the new tag"
for dir in k8s/base k8s/monitoring; do
  kubectl --context "$PROFILE" kustomize "$dir" \
    | sed -e "s#churn-api:dev#churn-api:$TAG#" -e "s#churn-frontend:dev#churn-frontend:$TAG#" \
    | kubectl --context "$PROFILE" apply -f -
done

echo "== Waiting for the rollout"
for d in api frontend alert-hub alertmanager prometheus grafana; do
  kubectl --context "$PROFILE" -n "$NS" rollout status "deploy/$d" --timeout=300s
done

echo "== Smoke test: the API answers and has a model"
for _ in $(seq 1 30); do
  health="$(kubectl --context "$PROFILE" -n "$NS" exec deploy/frontend -- wget -qO- http://api:8000/health || true)"
  [[ "$health" == *'"model_loaded":true'* ]] && break
  sleep 2
done
echo "$health"
[[ "$health" == *'"model_loaded":true'* ]] || { echo "API is up but has no model (is there a champion in MLflow?)"; exit 1; }

echo "== Smoke test: Prometheus scrapes every API pod and the alert hub answers"
for _ in $(seq 1 30); do
  targets="$(kubectl --context "$PROFILE" -n "$NS" exec deploy/frontend -- wget -qO- \
    'http://prometheus:9090/prometheus/api/v1/query?query=count(up%7Bjob%3D%22churn-api%22%7D%3D%3D1)' 2>/dev/null || true)"
  [[ "$targets" == *'"value":['*',"2"]'* || "$targets" == *'"value":['*',"3"]'* ]] && break
  sleep 2
done
echo "$targets"
[[ "$targets" == *'"value":['*',"2"]'* || "$targets" == *'"value":['*',"3"]'* ]] \
  || { echo "ERROR: Prometheus does not see the expected API pods up"; exit 1; }
kubectl --context "$PROFILE" -n "$NS" exec deploy/frontend -- wget -qO- http://alert-hub:8080/healthz

echo "== Removing old images (keeping the 3 newest besides $TAG)"
docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^churn-(api|frontend):' | grep -v -E ":($TAG|dev)$" \
  | tail -n +7 | xargs -r docker rmi >/dev/null 2>&1 || true
echo "Deployed $TAG"
