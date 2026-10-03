# Đóng góp và phân công

Tài liệu ghi vai trò và trách nhiệm của từng thành viên, cùng quy ước làm việc chung. Mỗi thành viên phải trả lời được các câu hỏi về phần mình khi bảo vệ.

## Phân công

| Thành viên | Mã SV | Vai trò | Trách nhiệm |
|---|---|---|---|
| Nguyễn Thị Hồng Hạnh | 25MS13316 | Logic bài toán và CI/CD | Nghiên cứu logic bài toán churn: bối cảnh, giả định kinh tế, metric. Thiết kế luồng CI/CD. |
| Chu Đức Bình | 25MS13303 | Kiến trúc MLOps | Thiết kế kiến trúc luồng MLOps, xác định các yêu cầu hệ thống cần đáp ứng, chọn tech stack và phân tích đánh đổi. |
| Phạm Văn Duy Khánh | 25MS13313 | Frontend, backend, serving | Viết code giao diện web, backend và serving API. |
| Nguyễn Phúc Lộc | 25MS13314 | Vận hành production, giám sát và cảnh báo | Triển khai môi trường production, vận hành (operation), giám sát (monitor) và cảnh báo (alert). |

### Sản phẩm và vị trí trong repo

| Thành viên | Sản phẩm | Nằm ở |
|---|---|---|
| Hạnh | Logic bài toán, giả định kinh tế, metric | `docs/open-questions.md` (Q1-Q5), `docs/spec-churn-training-pipeline.md` |
| Hạnh | Luồng CI/CD | `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`, `docs/rollout-strategies.md` |
| Bình | Kiến trúc và luồng dữ liệu, tech stack, trade-off | `ARCHITECTURE.md`, `docs/mlops-flow.html` |
| Bình | Yêu cầu hệ thống | `ARCHITECTURE.md` (mục 1), `README.md` (phần Bài toán) |
| Khánh | Frontend | `frontend/` |
| Khánh | Backend và serving | `src/serving/`, `deploy/Dockerfile.api`, `tests/test_api.py` |
| Lộc | Môi trường production (Kubernetes, Compose, runner) | `k8s/base/`, `docker-compose.yml`, `scripts/deploy-local.sh`, `run.sh` |
| Lộc | Giám sát và cảnh báo | `k8s/monitoring/` (Prometheus, Alertmanager, Grafana, Loki, Alloy), `src/alerts/hub.py` |
| Lộc | Hướng dẫn vận hành | `docs/deployment-guide.md`, `k8s/README.md` |

Một số hạng mục có liên quan chéo: phần giám sát cần Khánh xuất metric từ API (`src/serving/app.py`), và luồng deploy của Hạnh chạy qua script và runner của Lộc. Khi sửa phần của người khác, mở PR và gắn người đó review.

## Hạng mục chưa có người nhận

Các phần sau trong đề bài chưa nằm trong phân công trên và cần được gán trước khi nộp:

- Pipeline huấn luyện và thực nghiệm (`src/training/`, MLflow, DVC).
- Responsible AI: phân tích công bằng, SHAP/LIME, quyền riêng tư, đạo đức.
- Test: coverage, data quality, model validation.
- Slide và kịch bản demo.

## Quy ước làm việc

- Mỗi thay đổi làm trên một nhánh riêng, đặt tên theo loại việc (`docs/...`, `feat/...`, `fix/...`), rồi mở Pull Request vào `main`. Không commit thẳng vào `main`.
- PR cần CI xanh (lint, test, build) và ít nhất một thành viên khác xem qua trước khi merge.
- Không xóa nhánh sau khi merge.
- Commit message nêu rõ thay đổi gì và vì sao; mỗi người commit bằng tài khoản GitHub của chính mình để lịch sử phản ánh đúng đóng góp.
- Không commit bí mật: `.env` đã nằm trong `.gitignore`, chỉ commit `.env.example`.

## Chạy và kiểm tra trước khi mở PR

```bash
docker compose run --rm trainer ruff check src tests
docker compose run --rm trainer pytest -q
```

Hướng dẫn cài đặt đầy đủ ở [`README.md`](README.md), thiết kế ở [`ARCHITECTURE.md`](ARCHITECTURE.md), vận hành ở [`docs/deployment-guide.md`](docs/deployment-guide.md).
