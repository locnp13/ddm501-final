---
title: 'Pipeline huấn luyện churn (Telco) với MLflow, DVC, Evidently'
type: 'feature'
created: '2026-09-30'
status: 'done'
baseline_commit: 'NO_VCS'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Stack Compose đã chạy nhưng chưa có dữ liệu, code huấn luyện hay model trong MLflow, nên `/v1/predict` luôn trả 503.

**Approach:** Thêm pipeline tái lập được (DVC) tải và kiểm định dataset Telco, huấn luyện XGBoost có CV và Optuna, log MLflow kèm báo cáo Evidently, và chỉ đăng ký lên Production khi qua cổng chất lượng để API dự đoán được.

## Boundaries & Constraints

**Always:**
- Chạy trong container `trainer` (Compose profile `train`), không cài phụ thuộc lên Python máy host.
- Tách train/test phân tầng theo `Churn` (Telco không có cột thời gian), seed cố định, `TotalCharges` rỗng được ép về NaN rồi impute.
- Tối ưu PR-AUC (average precision) bằng CV 5-fold; xử lý mất cân bằng bằng `scale_pos_weight`, không dùng accuracy làm metric chính.
- Model đăng ký là `churn-model`, `predict` trả P(churn) dạng float cho mỗi dòng (API hiện tại đọc `result[0]`), stage `Production` chỉ khi PR-AUC test ≥ ngưỡng trong `params.yaml` (mặc định 0.50).
- Tham số (n_trials, test_size, seed, ngưỡng) nằm trong `params.yaml`; DVC dùng `dvc init --no-scm` vì chưa có git.

**Ask First:** `git init` hoặc thêm remote DVC; đổi ngưỡng cổng chất lượng; thêm cột nhạy cảm/fairness vào pipeline.

**Never:** fairness, SHAP/LIME, dashboard Grafana, CI, drift lúc serving (thuộc mục tiêu khác); sửa `src/serving/app.py` ngoài việc bổ sung phụ thuộc; xóa dữ liệu cũ trong `~/Downloads`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Chạy thành công | Có mạng, MLflow healthy | CSV vào `data/raw`, run MLflow có params/metrics/artifacts, model Production | N/A |
| Sai schema | Thiếu cột hoặc `Churn` ∉ {Yes,No} | Pandera báo lỗi, thoát mã ≠ 0 | Không tạo run, không đăng ký |
| Không đạt ngưỡng | PR-AUC test < ngưỡng | Run vẫn được log, model không lên Production | Thoát mã ≠ 0, in metric và ngưỡng |
| Mất mạng | Không tải được CSV | Dùng `data/raw` nếu đã có | Nếu chưa có: lỗi rõ ràng, thoát mã ≠ 0 |

</frozen-after-approval>

## Code Map

- `docker-compose.yml` -- thêm service `trainer`; MLflow ở `http://mlflow:5000`
- `deploy/Dockerfile.api`, `requirements.txt` -- image API nạp model nên cần cùng phụ thuộc (xgboost)
- `src/serving/app.py` -- nạp `models:/churn-model/Production`, đọc `result[0]` làm xác suất

## Tasks & Acceptance

**Execution:**
- [x] `requirements-train.txt`, `deploy/Dockerfile.train` -- xgboost, optuna, pandera, dvc, evidently, pytest (ghim phiên bản) -- môi trường huấn luyện
- [x] `requirements.txt` -- thêm `xgboost` (đúng phiên bản train) -- API nạp được model
- [x] `docker-compose.yml` -- service `trainer` (profile `train`, mount repo vào `/work`, đợi mlflow healthy) -- chạy `docker compose run --rm trainer dvc repro`
- [x] `src/training/data.py` -- tải CSV, schema Pandera, làm sạch, ghi `data/processed/churn.csv` -- ingest + validate
- [x] `src/training/model.py` -- wrapper pyfunc trả P(churn) -- khớp API
- [x] `src/training/train.py` -- split, Optuna + CV, đánh giá, báo cáo Evidently (train vs test), log MLflow, cổng chất lượng, đăng ký -- huấn luyện
- [x] `dvc.yaml`, `params.yaml`, `.dvc/` -- stage `ingest` và `train` -- tái lập
- [x] `tests/test_data.py`, `tests/test_gate.py` -- test ca sai schema, TotalCharges rỗng, quyết định cổng -- ma trận I/O
- [x] `CLAUDE.md` -- cập nhật lệnh thật (build, chạy pipeline, chạy test) -- tài liệu repo

**Acceptance Criteria:**
- Given stack đang chạy, when chạy `dvc repro` trong trainer, then có run MLflow chứa params, PR-AUC/ROC-AUC/recall/F1, báo cáo Evidently và model `churn-model` ở Production.
- Given model đã ở Production, when khởi động lại `api` và gọi `POST /v1/predict` với một dòng khách hàng hợp lệ, then nhận `churn_probability` trong [0,1] và `/health` báo `model_loaded: true`.
- Given không đổi dữ liệu và tham số, when chạy `dvc repro` lần hai, then DVC bỏ qua các stage đã cache.

## Spec Change Log

- 2026-09-30: nâng `xgboost` 2.1.3 -> 2.1.4 (image train và API) vì 2.1.3 không tương thích scikit-learn 1.6.0, khiến CV trả `nan`. Ghim thêm `multimethod==1.12`, `pathspec==0.12.1` cho pandera/DVC. Thêm MinIO làm artifact store và remote DVC. Các câu hỏi còn mở: `docs/open-questions.md`.

## Design Notes

Ghim phiên bản xgboost/sklearn giống nhau ở image train và API để tránh lỗi nạp model. Wrapper `predict` trả `predict_proba[:, 1]`, còn logic ngưỡng phân loại nằm ở tầng dùng kết quả, không nhúng vào model.

## Verification

**Commands:**
- `docker compose --profile train run --rm trainer pytest -q` -- expected: pass
- `docker compose --profile train run --rm trainer dvc repro` -- expected: exit 0, model Production
- `curl -s localhost:8000/health` sau `docker compose restart api` -- expected: `model_loaded: true`

**Manual checks (if no CLI):**
- Mở `http://localhost:5001`, xem run mới có báo cáo Evidently và bản đăng ký `churn-model` stage Production.
