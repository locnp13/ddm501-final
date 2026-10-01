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
             +-> /metrics -> Prometheus -> Grafana (alert rules)
```

Sơ đồ chi tiết: [`docs/mlops-flow.html`](docs/mlops-flow.html). Đặc tả pipeline huấn luyện: [`docs/spec-churn-training-pipeline.md`](docs/spec-churn-training-pipeline.md).

### Các service trong `docker-compose.yml`

| Service | Vai trò | Cổng host |
|---------|---------|-----------|
| `postgres` | Backend store của MLflow (metadata: run, params, metrics, registry) | - |
| `minio` | Object storage S3: bucket `mlflow` (artifact/model) và `dvc` (remote dữ liệu) | 9000 (S3), 9001 (console) |
| `mlflow` | Tracking server + Model Registry | 5001 |
| `api` | FastAPI phục vụ dự đoán | 8000 |
| `frontend` | Giao diện web (nginx, trang tĩnh): dự đoán, lịch sử, thông tin mô hình, tài liệu; proxy `/api` sang `api` | 8080 |
| `trainer` | Chạy pipeline huấn luyện (profile `train`) | - |
| `prometheus` | Thu thập metric, đánh giá alert rules | 9090 |
| `grafana` | Dashboard | 3000 |

MLflow dùng cổng 5001 vì cổng 5000 bị AirPlay Receiver chiếm trên macOS.

## Bắt đầu nhanh

Yêu cầu: Docker và Docker Compose. Không cần cài Python lên máy host.

```bash
# 0. Tạo khóa quản trị dùng khi duyệt model trên giao diện (tệp .env không được commit)
cp .env.example .env   # rồi đặt ADMIN_KEY, ví dụ: openssl rand -hex 24

# 1. Khởi động hạ tầng
docker compose up -d --build

# 2. Chạy pipeline: tải dữ liệu, kiểm định, huấn luyện, đăng ký model (alias challenger)
GIT_COMMIT=$(git rev-parse HEAD) GIT_DIRTY=$(git status --porcelain | wc -l) \
  docker compose run --rm trainer dvc repro

# 3. Duyệt model (bước của con người): mở http://localhost:8080, tab Mô hình, xem bảng so sánh rồi bấm Duyệt
#    (cần ADMIN_KEY trong .env). API nạp model mới ngay, không cần restart.
```

Sau đó:

- Giao diện web: http://localhost:8080 (tab Dự đoán, Lịch sử, Mô hình, Tài liệu)
- MLflow UI: http://localhost:5001
- Swagger UI: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (mặc định `minioadmin` / `minioadmin`, đổi bằng biến `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (tài khoản mặc định `admin` / `admin`, chỉ dùng cho môi trường phát triển)

### Ví dụ gọi API

```bash
curl -X POST http://localhost:8000/v1/predict \
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

## Giám sát

- Metric của API: `churn_requests_total`, `churn_request_latency_seconds`, `churn_predictions_total`, `churn_probability`, `churn_model_loaded`.
- Alert rules (`monitoring/alerts.yml`):

| Alert | Điều kiện | Mức |
|-------|-----------|-----|
| `ApiDown` | API không phản hồi 1 phút | critical |
| `ModelNotLoaded` | API chạy nhưng chưa có model, 2 phút | warning |
| `HighErrorRate` | Tỷ lệ 5xx > 5% trong 5 phút | critical |
| `HighP95Latency` | p95 latency > 500 ms trong 5 phút | warning |

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
deploy/           Dockerfile cho api, mlflow, trainer
monitoring/       Prometheus, alert rules, Grafana provisioning
tests/            test dữ liệu và quality gate
docs/             sơ đồ luồng MLOps, spec
```

## Trạng thái hiện tại

Pipeline huấn luyện đã chạy end-to-end (PR-AUC test 0.663, model `churn-model` v2 là champion, API dự đoán được). Các câu hỏi cần nhóm chốt: [`docs/open-questions.md`](docs/open-questions.md).

Đã có: pipeline huấn luyện, API cơ bản, stack Compose có healthcheck, metric và alert rules, CI lint/test/build.

Chưa hoàn thành (theo yêu cầu đề bài):

- [ ] Dashboard Grafana
- [ ] Test API (integration) và coverage > 80%
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
