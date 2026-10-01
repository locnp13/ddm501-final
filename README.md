# Customer Churn Prediction - DDM501 Final Project

[![CI](https://github.com/locnp13/ddm501-final/actions/workflows/ci.yml/badge.svg)](https://github.com/locnp13/ddm501-final/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

Hệ thống ML end-to-end dự đoán khách hàng có khả năng rời bỏ dịch vụ (churn), từ dữ liệu, huấn luyện, phục vụ qua API đến giám sát. Đồ án cuối môn **DDM501 - AI in Production** (Topic 3: Customer Churn Prediction).

## Nhóm

| Thành viên | Mã SV | Phụ trách |
|------------|-------|-----------|
| Nguyễn Thị Hồng Hạnh | 25MS13316 | _..._ |
| Nguyễn Phúc Lộc | 25MS13314 | _..._ |
| Phạm Văn Duy Khánh | 25MS13313 | _..._ |
| Chu Đức Bình | 25MS13303 | _..._ |

## Bài toán

- **Mục tiêu**: xác định khách hàng có nguy cơ churn để bộ phận chăm sóc khách hàng can thiệp sớm.
- **Dữ liệu**: [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (~7.000 khách hàng, ~26% churn).
- **Metric chính**: PR-AUC (average precision), vì lớp churn mất cân bằng. Không dùng accuracy làm metric chính. Ngoài ra theo dõi ROC-AUC, precision, recall, F1.
- **Mô hình**: XGBoost, tối ưu siêu tham số bằng Optuna, đánh giá chéo 5-fold, xử lý mất cân bằng bằng `scale_pos_weight`.

## Kiến trúc

```
DVC (ingest + validate) -> train (Optuna + CV) -> MLflow Tracking/Registry
   |  data/model-versioned in MinIO (S3)       metadata: Postgres | artifacts: MinIO
                                                        |
                                            quality gate (3 điều kiện) -> challenger -> duyệt -> champion
                                                        v
                                   models:/churn-model@champion
                                                        |
Client -> FastAPI /v1/predict  <-- nạp model ------------+
             |
             +-> /metrics -> Prometheus (trong cụm) -> Alertmanager -> alert-hub -> tab Cảnh báo
                                          +-> Grafana (3 dashboard)
```

Thiết kế đầy đủ (sơ đồ kiến trúc, luồng dữ liệu, lý do chọn công nghệ, đánh đổi): [`ARCHITECTURE.md`](ARCHITECTURE.md). Kiến trúc triển khai hiện tại (Compose + Kubernetes + runner) và hướng dẫn vận hành: [`docs/deployment-guide.md`](docs/deployment-guide.md). Sơ đồ luồng MLOps: [`docs/mlops-flow.html`](docs/mlops-flow.html). Đặc tả pipeline huấn luyện: [`docs/spec-churn-training-pipeline.md`](docs/spec-churn-training-pipeline.md).

### Các service trong `docker-compose.yml`

| Service | Vai trò | Cổng host |
|---------|---------|-----------|
| `postgres` | Backend store của MLflow (metadata: run, params, metrics, registry) | - |
| `minio` | Object storage S3: bucket `mlflow` (artifact/model) và `dvc` (remote dữ liệu) | 9000 (S3), 9001 (console) |
| `mlflow` | Tracking server + Model Registry | 5001 |
| `trainer` | Chạy pipeline huấn luyện (profile `train`) | - |

MLflow dùng cổng 5001 vì cổng 5000 bị AirPlay Receiver chiếm trên macOS.

`api`, `frontend` và toàn bộ giám sát (Prometheus, Alertmanager, Grafana, alert-hub) **không nằm trong Compose**: chúng chạy trên Kubernetes (minikube), xem phần Kubernetes bên dưới và [`k8s/README.md`](k8s/README.md).

## Bắt đầu nhanh

Yêu cầu: Docker, Docker Compose, minikube và kubectl. Không cần cài Python lên máy host.

```bash
# 0. Tạo khóa quản trị dùng khi duyệt model trên giao diện (tệp .env không được commit)
cp .env.example .env   # rồi đặt ADMIN_KEY và GRAFANA_ADMIN_PASSWORD, ví dụ: openssl rand -hex 24

# 1. Khởi động hạ tầng (MLflow, MinIO, Postgres)
docker compose up -d --build

# 1b. Dựng cụm Kubernetes và triển khai api + frontend (lần đầu, xem k8s/README.md), rồi mở cổng ra máy:
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8088:80
# 2. Chạy pipeline: tải dữ liệu, kiểm định, huấn luyện, đăng ký model (alias challenger)
GIT_COMMIT=$(git rev-parse HEAD) GIT_DIRTY=$(git status --porcelain | wc -l) \
  docker compose run --rm trainer dvc repro

# 3. Duyệt model (bước của con người): mở http://localhost:8088, tab Mô hình, xem bảng so sánh rồi bấm Duyệt
#    (cần ADMIN_KEY trong .env). API nạp model mới ngay, không cần restart.
```

Sau lần cài đặt đầu, mỗi lần dùng chỉ cần chạy `./run.sh`: script bật Docker Desktop, các container Compose và minikube nếu chưa chạy, deploy lên cụm nếu chưa có, mở cổng 8088 ra máy rồi mở các URL bên dưới trong trình duyệt. `./run.sh stop` tắt phần chuyển tiếp cổng (container và minikube vẫn chạy). Script không huấn luyện hay duyệt model; nếu chưa có `champion` nó chỉ nhắc.

Sau đó:

- Giao diện web: http://localhost:8088 (qua Ingress của minikube; tab Dự đoán, Lịch sử, Mô hình, Tài liệu)
- MLflow UI: http://localhost:5001
- Swagger UI: http://localhost:8088/api/docs
- MinIO console: http://localhost:9001 (mặc định `minioadmin` / `minioadmin`, đổi bằng biến `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`)
- Grafana: http://localhost:8088/grafana/ (tài khoản `admin`, mật khẩu là `GRAFANA_ADMIN_PASSWORD` trong `.env`)
- Prometheus: http://localhost:8088/prometheus/ · Alertmanager: http://localhost:8088/alertmanager/

### Ví dụ gọi API

```bash
curl -X POST http://localhost:8088/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 351.75
  }'
# ví dụ đầu ra: {"churn_probability": 0.71}
```

Khi chưa có model mang alias `champion`, `/v1/predict` trả `503 Model not loaded`. Đây là hành vi có chủ đích.

**Kiểm tra đầu vào** (schema Pydantic, ví dụ có sẵn trong Swagger): thiếu trường, sai kiểu hoặc số ngoài dải (`tenure` 0-120, `MonthlyCharges` >= 0, `SeniorCitizen` 0/1) trả `422` kèm trường lỗi; `TotalCharges` được phép `null` (khách mới). Giá trị phân loại chưa từng thấy khi huấn luyện (ví dụ `PaymentMethod: "Momo"`) vẫn được dự đoán, phản hồi có `warnings` và metric `churn_unknown_category_total` tăng. Phản hồi gồm `churn_probability`, `model_version`, `warnings`.

API còn có `GET /v1/model/versions` (lịch sử mọi phiên bản) và `POST /v1/model/rollback` (quay về bản cũ, cùng khóa quản trị; model được nạp và chạy thử trước khi đổi). Tab Mô hình của giao diện web có bảng lịch sử và nút Khôi phục. Cấu hình `MODEL_URI=models:/churn-model/3` ghim một phiên bản và `MODEL_POLL_SECONDS=30` cho API theo dõi alias; xem [`docs/rollout-strategies.md`](docs/rollout-strategies.md) (canary, A/B trên Kubernetes). API còn có `GET /v1/model/challenger` và `POST /v1/model/promote` (duyệt model: cần header `X-Admin-Key`, chỉ duyệt đúng bản đang là `challenger`, nạp thử model trước khi đổi alias, và phục vụ ngay không cần restart; tắt nếu `ADMIN_KEY` rỗng), `GET /v1/model` (phiên bản, chỉ số, cấu hình được chọn, nguồn gốc, đường cong lợi nhuận, so sánh cấu hình) và `GET /v1/model/figures/{calibration.png|profit_curve.png}`; giao diện web dùng các endpoint này.

## Pipeline huấn luyện

Hai stage DVC (`dvc.yaml`), tham số trong `params.yaml`:

1. **`ingest`** (`src/training/data.py`): tải CSV, kiểm định schema bằng Pandera, ép `TotalCharges` rỗng về NaN, ghi `data/processed/churn.csv`.
2. **`train`** (`src/training/train.py`): chia train/test phân tầng theo `Churn`; so sánh Dummy, Logistic Regression, Random Forest, XGBoost với/không có feature mới và với/không có cột nhạy cảm bằng CV (mỗi cấu hình là một run MLflow); tune XGBoost bằng Optuna; hiệu chuẩn xác suất (isotonic); tính lợi nhuận và Recall@k theo mức liên hệ khách; ghi params/metrics/artifact và provenance (git commit, hash dữ liệu) lên MLflow; sinh báo cáo Evidently (train vs test, chỉ là sanity check); kiểm tra quality gate. Chọn mô hình chỉ dựa vào CV, tập test dùng một lần.

**Quality gate** (3 điều kiện, tham số trong `params.yaml`): PR-AUC test >= sàn 0.50; không tệ hơn baseline Logistic Regression (bỏ qua nếu chính LogReg thắng); không tệ hơn champion hiện tại quá 0.005. Qua gate thì model được đăng ký với alias `challenger` và **chưa được phục vụ**; hệ thống không bao giờ tự duyệt. Một thành viên duyệt trên giao diện web (tab Mô hình, nút Duyệt, cần `ADMIN_KEY`) hoặc bằng `python -m src.training.promote`, khi đó model thành `champion` và API phục vụ nó. Nếu không qua, run vẫn được log, model không được đăng ký và tiến trình thoát với mã khác 0.

### Dữ liệu và remote DVC (MinIO)

Dữ liệu được version bằng DVC và lưu ở bucket `dvc` trên MinIO (cấu hình trong `.dvc/config`).

```bash
docker compose run --rm trainer dvc push   # đẩy dữ liệu lên MinIO
docker compose run --rm trainer dvc pull   # lấy dữ liệu đúng phiên bản của commit hiện tại
```

Image MinIO chính thức không còn được phát hành công khai, nên compose dùng `bitnamilegacy/minio:2024.12.18` (bản MinIO đóng băng, không nhận bản vá). Phù hợp đồ án, không nên dùng cho production.

## Kubernetes (local)

`api`, `frontend` và bộ giám sát chạy trên minikube, kèm canary theo trọng số; hướng dẫn và các lưu ý ở [`k8s/README.md`](k8s/README.md). Deploy bằng nút **Run workflow** trên GitHub (tab Actions, workflow *Deploy to local Kubernetes*): kiểm tra runner trên máy đang chạy, rồi build image arm64 và triển khai lên cụm; xem phần Deploy trong `k8s/README.md`.

## Giám sát

Chạy trong cụm Kubernetes, cấu hình trong [`k8s/monitoring/`](k8s/monitoring/):

| Thành phần | Vai trò | Truy cập |
|---|---|---|
| Prometheus | Quét **từng pod API** (đọc annotation `prometheus.io/*`), đánh giá luật cảnh báo, giữ dữ liệu 7 ngày | `/prometheus/` |
| Alertmanager | Gom nhóm cảnh báo và gửi bằng webhook | `/alertmanager/` |
| alert-hub | Nhận webhook, giữ danh sách cảnh báo, phục vụ tab **Cảnh báo** của giao diện (có chấm đỏ) | tab Cảnh báo |
| Grafana | 3 dashboard: **Churn: API**, **Churn: Mô hình**, **Churn: So sánh phiên bản** | `/grafana/` |

Số liệu của API: `churn_requests_total`, `churn_request_latency_seconds` (cả hai có nhãn `model_version`), `churn_predictions_total`, `churn_probability`, `churn_model_info`, `churn_model_loaded`, `churn_model_changes_total`, `churn_unknown_category_total`, `churn_feature_values_total` và `churn_feature_baseline_share` (drift đầu vào, xem bên dưới), cùng số liệu tiến trình (`process_cpu_seconds_total`, `process_resident_memory_bytes`).

Luật cảnh báo (`k8s/monitoring/prometheus/alerts.yml`, được kiểm thử bằng `promtool test rules` trong CI). Ngưỡng là điểm khởi đầu của dự án, chưa đo từ dữ liệu thật:

| Cảnh báo | Điều kiện | Mức |
|---|---|---|
| `ApiNoTargets` | Prometheus không thấy pod API nào, 2 phút | critical |
| `ApiDown` | Một pod API không phản hồi, 1 phút | critical |
| `HighErrorRate` | Lỗi 5xx > 5% trong 5 phút | critical |
| `ReplicaLost` | Dưới 2 pod stable đang chạy, 3 phút | warning |
| `ModelNotLoaded` | Pod chạy nhưng chưa có model, 2 phút | warning |
| `HighValidationErrorRate` | Yêu cầu bị từ chối (422) > 20% trong 10 phút | warning |
| `HighP95Latency` | p95 độ trễ > 500 ms trong 5 phút | warning |
| `HighMemory` | RAM pod > 600 MiB (giới hạn 768 MiB), 5 phút | warning |
| `UnknownCategorySpike` | Giá trị phân loại chưa từng thấy > 5% số dự đoán trong 15 phút | warning |
| `PredictionDrift` | Xác suất trung bình 1 giờ lệch quá 0,10 so với 0,27 (tỷ lệ churn huấn luyện), kéo dài 30 phút, từ 50 dự đoán trở lên | warning |
| `FeatureDrift` | PSI của một feature so với dữ liệu huấn luyện > 0,2 trong 1 giờ, kéo dài 30 phút (cần trên 100 dự đoán) | warning |
| `ModelChanged` | Model vừa được duyệt, khôi phục hoặc đồng bộ (10 phút) | info |

**Drift theo từng feature (PSI).** Khi huấn luyện, phân phối của từng đầu vào (19 feature, số liệu liên tục chia theo thập phân vị) được lưu cùng model (`feature_baseline.json`). API đếm mỗi yêu cầu vào đúng các nhóm đó và Prometheus tính PSI so với lúc huấn luyện: dưới 0,1 ổn định, 0,1 đến 0,2 lệch vừa, trên 0,2 lệch đáng kể. Chỉ có **số đếm theo nhóm**, không lưu giá trị của khách hàng nào. Model huấn luyện trước tính năng này cần bổ sung thống kê: `docker compose run --rm trainer python -m src.training.baseline` (mặc định cho champion; từ chối nếu dữ liệu đã đổi). Panel PSI nằm trong dashboard *Churn: Mô hình*.

Giám sát không đo được độ chính xác thật của model vì nhãn churn đến muộn; `PredictionDrift`, `UnknownCategorySpike` và `FeatureDrift` chỉ báo hiệu dữ liệu hoặc dự đoán đã đổi.

## Kiểm thử và CI

```bash
docker compose run --rm trainer pytest          # unit + data-quality + model-gate
```

GitHub Actions (`.github/workflows/ci.yml`) chạy khi push lên `main` và khi mở PR: `ruff` -> `pytest` với coverage -> build image API (đẩy lên GHCR khi merge vào `main`).

## Cấu trúc thư mục

```
src/training/     ingest, validate, train, wrapper model
src/serving/      FastAPI app, thông tin model cho giao diện
frontend/         giao diện web (nginx + HTML/CSS/JS thuần)
k8s/              manifest minikube: api, frontend, Ingress, canary
k8s/monitoring/   Prometheus, Alertmanager, Grafana (dashboard), alert-hub, luật cảnh báo và test luật
deploy/           Dockerfile cho api, mlflow, trainer
src/alerts/       alert-hub: nhận webhook của Alertmanager, phục vụ tab Cảnh báo
tests/            test dữ liệu và quality gate
docs/             hướng dẫn triển khai và vận hành, sơ đồ luồng MLOps, spec, quyết định thiết kế
```

## Trạng thái hiện tại

Pipeline huấn luyện đã chạy end-to-end (PR-AUC test 0.663, model `churn-model` v3 là champion, API chạy trên Kubernetes). Các câu hỏi cần nhóm chốt: [`docs/open-questions.md`](docs/open-questions.md).

Đã có: pipeline huấn luyện, API có kiểm tra đầu vào và duyệt/khôi phục model, giao diện web, giám sát (Prometheus, Alertmanager, 3 dashboard Grafana, 11 luật cảnh báo có test) trên Kubernetes, CI lint/test/build/kiểm tra cấu hình giám sát, test API (coverage khoảng 90%).

Chưa hoàn thành (theo yêu cầu đề bài):

- [ ] Kênh gửi cảnh báo ra ngoài (Slack, email): hiện cảnh báo chỉ vào tab Cảnh báo của giao diện
- [ ] Drift theo từng feature (PSI) và nhật ký dự đoán
- [ ] Endpoint `/v1/predict/batch` (schema chặt và ví dụ OpenAPI đã có cho `/v1/predict`)
- [ ] Giải thích mô hình (SHAP, LIME)
- [ ] Phân tích fairness và giảm thiểu thiên lệch
- [ ] Tài liệu privacy, ethics, hướng dẫn vận hành, so sánh lựa chọn công nghệ
- [ ] Bước deploy trong CI

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|-------------|------------------------|------------|
| `/v1/predict` trả 503 | Chưa có model `champion` hoặc chưa duyệt model | Chạy `dvc repro`, rồi duyệt model ở tab Mô hình (hoặc `python -m src.training.promote` và `docker compose restart api`) |
| `dvc repro` thoát mã khác 0 ở bước train | Không qua quality gate (xem tag `gate_failures` của run) | Xem metric ở MLflow, chỉnh `params.yaml` hoặc mô hình |
| Lỗi tải dữ liệu | Không có mạng và chưa có `data/raw/telco.csv` | Kết nối mạng, hoặc đặt CSV vào `data/raw/` |
| Không mở được MLflow ở cổng 5000 | macOS AirPlay Receiver chiếm cổng | Dùng http://localhost:5001 |
| `dvc push` báo lỗi kết nối | Chạy ngoài Docker nên không phân giải được `minio` | Chạy qua `docker compose run --rm trainer ...` |
| Cổng đã bị chiếm | Dịch vụ khác đang chạy | Đổi ánh xạ cổng trong `docker-compose.yml` |
