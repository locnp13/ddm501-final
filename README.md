# Customer Churn Prediction - DDM501 Final Project

[![CI](https://github.com/locnp13/ddm501-final/actions/workflows/ci.yml/badge.svg)](https://github.com/locnp13/ddm501-final/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

Hệ thống ML end-to-end dự đoán khách hàng có khả năng rời bỏ dịch vụ (churn), từ dữ liệu, huấn luyện, phục vụ qua API đến giám sát. Đồ án cuối môn **DDM501 - AI in Production** (Topic 3: Customer Churn Prediction).

## Nhóm

| Thành viên | Mã SV | Phụ trách |
|------------|-------|-----------|
| _Họ tên 1_ | _..._ | _..._ |
| _Họ tên 2_ | _..._ | _..._ |
| _Họ tên 3_ | _..._ | _..._ |

## Bài toán

- **Mục tiêu**: xác định khách hàng có nguy cơ churn để bộ phận chăm sóc khách hàng can thiệp sớm.
- **Dữ liệu**: [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (~7.000 khách hàng, ~26% churn).
- **Metric chính**: PR-AUC (average precision), vì lớp churn mất cân bằng. Không dùng accuracy làm metric chính. Ngoài ra theo dõi ROC-AUC, precision, recall, F1.
- **Mô hình**: XGBoost, tối ưu siêu tham số bằng Optuna, đánh giá chéo 5-fold, xử lý mất cân bằng bằng `scale_pos_weight`.

## Kiến trúc

```
DVC (ingest + validate) -> train (Optuna + CV) -> MLflow Tracking/Registry
                                                        |
                                            quality gate (PR-AUC >= ngưỡng)
                                                        v
                                   models:/churn-model/Production
                                                        |
Client -> FastAPI /v1/predict  <-- nạp model ------------+
             |
             +-> /metrics -> Prometheus -> Grafana (alert rules)
```

Sơ đồ chi tiết: [`docs/mlops-flow.html`](docs/mlops-flow.html). Đặc tả pipeline huấn luyện: [`docs/spec-churn-training-pipeline.md`](docs/spec-churn-training-pipeline.md).

### Các service trong `docker-compose.yml`

| Service | Vai trò | Cổng host |
|---------|---------|-----------|
| `postgres` | Backend store của MLflow | - |
| `mlflow` | Tracking server + Model Registry | 5001 |
| `api` | FastAPI phục vụ dự đoán | 8000 |
| `trainer` | Chạy pipeline huấn luyện (profile `train`) | - |
| `prometheus` | Thu thập metric, đánh giá alert rules | 9090 |
| `grafana` | Dashboard | 3000 |

MLflow dùng cổng 5001 vì cổng 5000 bị AirPlay Receiver chiếm trên macOS.

## Bắt đầu nhanh

Yêu cầu: Docker và Docker Compose. Không cần cài Python lên máy host.

```bash
# 1. Khởi động hạ tầng
docker compose up -d --build

# 2. Chạy pipeline: tải dữ liệu, kiểm định, huấn luyện, đăng ký model
docker compose run --rm trainer dvc repro

# 3. Nạp lại model trong API (API chỉ nạp model lúc khởi động)
docker compose restart api
```

Sau đó:

- MLflow UI: http://localhost:5001
- Swagger UI: http://localhost:8000/docs
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

Khi chưa có model ở stage `Production`, `/v1/predict` trả `503 Model not loaded`. Đây là hành vi có chủ đích.

## Pipeline huấn luyện

Hai stage DVC (`dvc.yaml`), tham số trong `params.yaml`:

1. **`ingest`** (`src/training/data.py`): tải CSV, kiểm định schema bằng Pandera, ép `TotalCharges` rỗng về NaN, ghi `data/processed/churn.csv`.
2. **`train`** (`src/training/train.py`): chia train/test phân tầng theo `Churn`, tối ưu bằng Optuna với CV, ghi params/metrics/artifact lên MLflow, sinh báo cáo drift Evidently (train vs test), kiểm tra quality gate.

**Quality gate**: model chỉ được chuyển lên `Production` khi PR-AUC trên tập test đạt `gate.min_pr_auc` (mặc định 0.50). Nếu không đạt, run vẫn được log nhưng model không được đăng ký và tiến trình thoát với mã khác 0.

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
src/serving/      FastAPI app
deploy/           Dockerfile cho api, mlflow, trainer
monitoring/       Prometheus, alert rules, Grafana provisioning
tests/            test dữ liệu và quality gate
docs/             sơ đồ luồng MLOps, spec
```

## Trạng thái hiện tại

Đã có: pipeline huấn luyện, API cơ bản, stack Compose có healthcheck, metric và alert rules, CI lint/test/build.

Chưa hoàn thành (theo yêu cầu đề bài):

- [ ] Dashboard Grafana
- [ ] Test API (integration) và coverage > 80%
- [ ] Endpoint batch, request/response schema chặt và OpenAPI examples
- [ ] Giải thích mô hình (SHAP, LIME)
- [ ] Phân tích fairness và giảm thiểu thiên lệch
- [ ] Tài liệu privacy, ethics, hướng dẫn vận hành, so sánh lựa chọn công nghệ
- [ ] Bước deploy trong CI

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|-------------|------------------------|------------|
| `/v1/predict` trả 503 | Chưa có model `Production` hoặc API chưa nạp lại | Chạy `dvc repro`, rồi `docker compose restart api` |
| `dvc repro` thoát mã khác 0 ở bước train | PR-AUC dưới ngưỡng gate | Xem metric ở MLflow, chỉnh `params.yaml` hoặc mô hình |
| Lỗi tải dữ liệu | Không có mạng và chưa có `data/raw/telco.csv` | Kết nối mạng, hoặc đặt CSV vào `data/raw/` |
| Không mở được MLflow ở cổng 5000 | macOS AirPlay Receiver chiếm cổng | Dùng http://localhost:5001 |
| Cổng đã bị chiếm | Dịch vụ khác đang chạy | Đổi ánh xạ cổng trong `docker-compose.yml` |
