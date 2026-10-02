# Kubernetes trên máy local (minikube)

Tổng quan kiến trúc, luồng deploy và vận hành: [`docs/deployment-guide.md`](../docs/deployment-guide.md). Tài liệu này là phần chi tiết về cụm.

Chỉ `api` và `frontend` chạy trong cụm. Postgres, MinIO và MLflow vẫn chạy bằng Docker Compose trên máy (Compose không còn `api` và `frontend`); pod gọi MLflow qua `host.minikube.internal:5001`, nên **stack Compose phải đang chạy** (`docker compose up -d mlflow minio postgres`).

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
kubectl -n churn create secret generic churn-secrets --from-env-file=.env   # cần ADMIN_KEY và GRAFANA_ADMIN_PASSWORD; TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID là tùy chọn
kubectl apply -k k8s/base
kubectl apply -k k8s/monitoring    # Prometheus, Alertmanager, Grafana, alert-hub
kubectl -n churn rollout status deploy/api

# Mở giao diện: http://localhost:8088
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8088:80
```

Cập nhật code bằng tay: `scripts/deploy-local.sh` (build image arm64 vào Docker của cụm, gắn tag theo commit, áp manifest, đợi cuốn bản mới và kiểm tra API có model).

## Deploy bằng nút trên GitHub

Actions tab, workflow **Deploy to local Kubernetes**, **Run workflow**, chọn nhánh. Push code **không** tự deploy; CI (`lint-test`, `build`) chỉ kiểm tra.

```
Run workflow ─┬─ runner-check (GitHub): chờ tối đa 60s để runner nhận job; không có thì hủy run và báo lỗi rõ
              └─ deploy (runner trên Mac, nhãn churn-local):
                   kiểm tra Docker và MLflow -> build image arm64 vào Docker của cụm -> kubectl apply
                   -> đợi cuốn bản mới -> smoke test (API có model)
```

- **Kiểm tra runner**: job cho runner offline sẽ nằm chờ tới 24 giờ, nên có một job song song trên GitHub theo dõi xem runner đã nhận `deploy` chưa; sau 60 giây chưa nhận thì hủy run với thông báo cách bật lại runner.
- **Image được build lại trên runner**, không kéo từ GHCR: image CI build là amd64, còn cụm chạy arm64. Image chạy trên cụm vì vậy không phải đúng tệp CI đã đẩy lên GHCR.
- **Điều kiện**: Mac bật và đăng nhập (runner là LaunchAgent), Docker Desktop đang chạy, và Compose đang chạy MLflow, MinIO, Postgres. Thiếu Docker hoặc MLflow thì bước kiểm tra trong script báo lỗi ngay. Nếu cụm đang tắt, script tự `minikube start`.
- **Nút này triển khai bất kể CI của commit đó xanh hay đỏ.** Nên chờ CI xanh trước khi bấm.
- **Secret `churn-secrets` tạo tay một lần** (xem phần Dựng cụm); script không đụng tới nó.
- **Cuốn bản mới an toàn**: pod chỉ nhận lưu lượng khi `/ready` trả 200, tức là đã nạp xong model.
- **Bảo mật**: chỉ người có quyền ghi mới bấm được; PR (kể cả từ fork) không chạy được workflow này. PR từ người ngoài phải được duyệt trước khi chạy CI (`all_external_contributors`). `main` chưa bật bảo vệ nhánh.
- Chạy tay không qua GitHub: `scripts/deploy-local.sh`.

Quản lý runner (cài ở `~/actions-runner-ddm501`):

```bash
cd ~/actions-runner-ddm501
./svc.sh status | stop | start
./svc.sh stop && ./svc.sh uninstall                      # gỡ dịch vụ
./config.sh remove --token "$(gh api -X POST repos/locnp13/ddm501-final/actions/runners/registration-token --jq .token)"
```

## Cấu trúc

| Thành phần | Vai trò |
|---|---|
| `Deployment/api` (2 bản sao) | Phục vụ model `@champion`, `MODEL_POLL_SECONDS=30` để các bản sao đồng bộ sau khi duyệt hoặc khôi phục |
| `Deployment/frontend` | Giao diện web (nginx) |
| `Ingress` | Trình duyệt chỉ nói chuyện với Ingress: `/api/*` đi vào API (bỏ tiền tố `/api`), còn lại vào giao diện. Ingress không ràng buộc host nên mở được bằng `localhost` |
| `k8s/monitoring/` | Prometheus (có PVC 2 GiB, giữ 7 ngày), Alertmanager, Grafana, alert-hub, Role chỉ đọc pod trong namespace `churn` cho Prometheus. Truy cập qua `/prometheus/`, `/alertmanager/`, `/grafana/`; webhook của hub **không** mở ra ngoài, Ingress chỉ cho `GET /hub/alerts` |
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
- **Image là bản build local** (`churn-api:<commit>`, `imagePullPolicy: IfNotPresent`), không phải image trên GHCR. Luồng tự deploy xem phần trên.
- **Prometheus trong cụm quét từng pod** (kể cả pod canary, nhãn `track=canary`) nhờ annotation `prometheus.io/*`; dashboard "So sánh phiên bản" tách số liệu theo `model_version`.
- ConfigMap của giám sát được sinh bằng kustomize và phải khai báo `namespace: churn`, nếu không chúng rơi vào namespace `default` và pod không tìm thấy (đã gặp khi triển khai).
- Dữ liệu Prometheus nằm trong PVC của minikube: mất nếu xóa cụm. Trạng thái Alertmanager (silence) và thay đổi trong giao diện Grafana nằm trong `emptyDir`, mất khi pod khởi động lại; dashboard thì lấy từ repo nên luôn được dựng lại.
- `kubectl config` chuyển sang context `churn`. Context `minikube` cũ trong kubeconfig trỏ tới cụm không còn tồn tại và không bị đụng tới.
