# Kubernetes trên máy local (minikube)

Chỉ `api` và `frontend` chạy trong cụm. Postgres, MinIO và MLflow vẫn chạy bằng Docker Compose trên máy; pod gọi MLflow qua `host.minikube.internal:5001`, nên **stack Compose phải đang chạy** (`docker compose up -d mlflow minio postgres`).

## Dựng cụm

```bash
minikube start -p churn --driver=docker --cpus=4 --memory=4096     # profile riêng, không đụng cụm khác
minikube -p churn addons enable ingress

# Build image trực tiếp vào Docker của cụm (không cần registry)
eval "$(minikube -p churn docker-env)"
docker build -t churn-api:dev -f deploy/Dockerfile.api .
docker build -t churn-frontend:dev -f frontend/Dockerfile frontend

# Khóa quản trị lấy từ .env (không có trong manifest), rồi triển khai
kubectl apply -f k8s/base/namespace.yaml
kubectl -n churn create secret generic churn-secrets --from-env-file=.env
kubectl apply -k k8s/base
kubectl -n churn rollout status deploy/api

# Mở giao diện: http://localhost:8088
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8088:80
```

Cập nhật code: build lại image (cùng tag `:dev`) rồi `kubectl -n churn rollout restart deploy/api deploy/frontend`.

## Cấu trúc

| Thành phần | Vai trò |
|---|---|
| `Deployment/api` (2 bản sao) | Phục vụ model `@champion`, `MODEL_POLL_SECONDS=30` để các bản sao đồng bộ sau khi duyệt hoặc khôi phục |
| `Deployment/frontend` | Giao diện web (nginx) |
| `Ingress` | Trình duyệt chỉ nói chuyện với Ingress: `/api/*` đi vào API (bỏ tiền tố `/api`), còn lại vào giao diện. Ingress không ràng buộc host nên mở được bằng `localhost` |
| `Service/api-admin` | Cùng các pod stable, dùng riêng cho `/api/v1/model/promote` và `/rollback` (xem bên dưới) |

## Canary

```bash
kubectl apply -k k8s/canary                  # api-canary ghim models:/churn-model/2, nhận 20% lưu lượng /api
kubectl -n churn annotate ingress api-canary nginx.ingress.kubernetes.io/canary-weight=50 --overwrite
kubectl delete -k k8s/canary                 # gỡ canary
```

Đổi phiên bản canary bằng biến `MODEL_URI` trong `k8s/canary/api-canary.yaml`. Đã kiểm tra: 200 dự đoán qua Ingress với trọng số 20% cho 43 yêu cầu vào canary (v2) và 157 vào stable (v3).

## Những điều rút ra khi kiểm tra

- **ingress-nginx chia canary theo Service, không theo đường dẫn.** Mọi Ingress trỏ tới Service `api` đều bị chia, kể cả đường dẫn quản trị. Canary là bản ghim nên từ chối duyệt/khôi phục, nên lúc đầu khoảng 20% lệnh duyệt bị từ chối. Vì vậy lệnh quản trị đi qua Service `api-admin` riêng; sau khi sửa, 120/120 yêu cầu quản trị vào đúng nhóm stable.
- **Image là bản build local** (`churn-api:dev`, `imagePullPolicy: IfNotPresent`), không phải image trên GHCR. Chưa có luồng tự deploy: xem `docs/rollout-strategies.md`.
- **Prometheus của Compose chưa scrape pod trong cụm.** Pod có annotation `prometheus.io/*` nhưng chưa có gì đọc chúng.
- `kubectl config` chuyển sang context `churn`. Context `minikube` cũ trong kubeconfig trỏ tới cụm không còn tồn tại và không bị đụng tới.
