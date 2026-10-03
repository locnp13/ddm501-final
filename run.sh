#!/usr/bin/env bash
# Bring the whole stack up if it is not already running, forward the ingress port and open the UIs.
#   ./run.sh         start whatever is down, then open the browser
#   ./run.sh stop    stop the port-forward (containers and minikube are left running)
# Steps skipped when already done: Docker Desktop, docker compose services, minikube, deploy to the
# cluster. First run also builds and deploys the images (a few minutes) and needs .env (see .env.example).
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.docker/bin:$PATH"
cd "$(dirname "$0")"

PROFILE=churn
NS=churn
PORT=8088
BASE="http://localhost:$PORT"
PID_FILE="${TMPDIR:-/tmp}/churn-port-forward.pid"
LOG_FILE="${TMPDIR:-/tmp}/churn-port-forward.log"

stop_forward() {
  if [[ -f "$PID_FILE" ]]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  stop_forward
  echo "Port-forward stopped. Containers and minikube keep running (minikube stop -p $PROFILE to free memory)."
  exit 0
fi

step() { printf '\n== %s\n' "$*"; }

# `docker info` hangs instead of failing while Docker Desktop is starting or wedged, and macOS has no timeout(1).
docker_ok() {
  docker info >/dev/null 2>&1 &
  local pid=$! i
  for i in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || { wait "$pid"; return; }
    sleep 1
  done
  kill "$pid" 2>/dev/null || true
  return 1
}

step "Docker"
if ! docker_ok; then
  echo "Docker is not answering; starting Docker Desktop"
  open -a Docker
  for _ in $(seq 1 12); do docker_ok && break; sleep 5; done
  docker_ok || { echo "ERROR: the Docker engine does not answer after 3 minutes. Quit and reopen Docker Desktop (or restart it) and run again."; exit 1; }
fi
echo "docker: ok"

step "Compose services (postgres, minio, mlflow)"
[[ -f .env ]] || { echo "ERROR: .env is missing (cp .env.example .env and set the keys)"; exit 1; }
docker compose up -d --wait postgres minio mlflow

step "minikube"
if minikube -p "$PROFILE" status >/dev/null 2>&1; then
  echo "cluster '$PROFILE' is running"
else
  minikube start -p "$PROFILE" --driver=docker --cpus=4 --memory=4096
fi
minikube -p "$PROFILE" addons enable ingress >/dev/null 2>&1 || true

step "Cluster workloads"
if ! kubectl --context "$PROFILE" get ns "$NS" >/dev/null 2>&1; then
  kubectl --context "$PROFILE" apply -f k8s/base/namespace.yaml
fi
if ! kubectl --context "$PROFILE" -n "$NS" get secret churn-secrets >/dev/null 2>&1; then
  echo "Creating secret churn-secrets from .env"
  kubectl --context "$PROFILE" -n "$NS" create secret generic churn-secrets --from-env-file=.env
fi
if kubectl --context "$PROFILE" -n "$NS" get deploy api frontend grafana >/dev/null 2>&1; then
  echo "already deployed; waiting for pods"
  for d in api frontend alert-hub alertmanager prometheus grafana loki alloy; do
    kubectl --context "$PROFILE" -n "$NS" rollout status "deploy/$d" --timeout=300s
  done
else
  echo "not deployed yet; running scripts/deploy-local.sh"
  scripts/deploy-local.sh
fi

step "Port-forward $PORT -> ingress"
# rollout status can report success while the restarted controller pod is still not Ready.
kubectl --context "$PROFILE" -n ingress-nginx wait --for=condition=Ready pod \
  -l app.kubernetes.io/component=controller --timeout=180s
if ! curl -fsS --max-time 3 -o /dev/null "$BASE/"; then
  stop_forward
  nohup kubectl --context "$PROFILE" -n ingress-nginx port-forward \
    svc/ingress-nginx-controller "$PORT:80" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  for _ in $(seq 1 20); do curl -fsS --max-time 3 -o /dev/null "$BASE/" && break; sleep 1; done
fi
curl -fsS --max-time 3 -o /dev/null "$BASE/" || { echo "ERROR: $BASE does not answer (see $LOG_FILE)"; exit 1; }
echo "$BASE: ok"

health="$(curl -fsS --max-time 5 "$BASE/api/health" || true)"
echo "api health: $health"
[[ "$health" == *'"model_loaded":true'* ]] \
  || echo "NOTE: no champion model yet. Train (docker compose run --rm trainer dvc repro), then approve it in the Mô hình tab."

step "Opening URLs"
for url in "$BASE/" "$BASE/api/docs" "$BASE/grafana/" "$BASE/prometheus/" "$BASE/alertmanager/" "http://localhost:5001" "http://localhost:9001"; do
  echo "$url"
  open "$url"
done
echo
echo "Done. Stop the port-forward with ./run.sh stop"
