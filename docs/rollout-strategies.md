# Quay về bản cũ, canary và A/B testing

Tài liệu này mô tả cách hệ thống hiện tại hỗ trợ quay về phiên bản cũ, và cách triển khai canary hoặc A/B trên Kubernetes. Manifest cho minikube nằm trong `k8s/` (xem `k8s/README.md`): **canary đã chạy thử trên minikube**; **A/B chưa làm**, chỉ là thiết kế.

## 1. Hiện tại (Docker Compose)

Một container `api` phục vụ model mang alias `champion`. Mọi phiên bản đã đăng ký đều được giữ lại trong MLflow (xem `docs/open-questions.md`, mục kết quả mảng 1).

| Việc | Cách làm |
|---|---|
| Xem các phiên bản | Tab Mô hình, bảng "Lịch sử phiên bản" (`GET /v1/model/versions`) |
| Duyệt model mới | Nút Duyệt (`POST /v1/model/promote`), chỉ nhận đúng bản đang là `challenger` |
| Quay về bản cũ | Nút Khôi phục (`POST /v1/model/rollback`), cần `ADMIN_KEY` |

Trước khi đổi, server nạp model đích và chạy thử một dự đoán mẫu theo schema API hiện tại. Nạp lỗi hoặc không tương thích thì **không có gì thay đổi**. Mỗi lần đổi ghi tag `approved_at` hoặc `restored_at` lên phiên bản đó và tăng `churn_model_changes_total{kind=...}`.

Lưu ý khi khôi phục: v1 được huấn luyện **trước khi có bước hiệu chuẩn xác suất**, nên điểm của nó cao hơn xác suất thật (cùng khách mẫu ra 0.85 so với 0.77). Giao diện tính mức rủi ro và lợi nhuận dựa trên giả định xác suất đã hiệu chuẩn, nên với v1 các con số đó không còn đúng.

## 2. Những gì đã chuẩn bị cho Kubernetes

| Khả năng | Cấu hình | Dùng để |
|---|---|---|
| **Ghim một phiên bản** | `MODEL_URI=models:/churn-model/3` | Chạy hai phiên bản song song. Instance ghim không đổi model và từ chối duyệt/khôi phục (409) |
| **Theo dõi alias** | `MODEL_POLL_SECONDS=30` (mặc định 0 = tắt) | Nhiều bản sao cùng hội tụ về `champion` sau một lần duyệt hoặc khôi phục. Bản sao nhận lệnh đổi ngay, các bản còn lại đổi trong tối đa 30 giây |
| **Nhận diện phiên bản** | `model_version` trong phản hồi; nhãn `model_version` trên `churn_predictions_total` và `churn_probability`; gauge `churn_model_info{version}` | Gán mỗi dự đoán và mỗi số liệu cho đúng phiên bản |

## 3. Canary (đã chạy thử trên minikube)

Mục tiêu: kiểm tra phiên bản mới chạy ổn định với một phần nhỏ lưu lượng trước khi chuyển hẳn.

- Đã kiểm tra bằng `k8s/canary/`: trọng số 20% cho 21,5% yêu cầu vào canary. Lưu ý: ingress-nginx chia canary **theo Service**, nên lệnh duyệt/khôi phục phải đi qua Service riêng (`api-admin`) để không rơi vào bản ghim (xem `k8s/README.md`).
- Hai Deployment dùng cùng image API: `churn-api-stable` ghim phiên bản hiện tại (ví dụ `models:/churn-model/3`) và `churn-api-canary` ghim phiên bản mới (`models:/churn-model/4`).
- Chia lưu lượng ở tầng Ingress hoặc service mesh (ingress-nginx với annotation canary-weight, Argo Rollouts hoặc Istio VirtualService), ví dụ 5% rồi 25% rồi 100%.
- So sánh hai phiên bản trên Grafana, ví dụ:
  - Tỷ lệ dự đoán churn theo phiên bản: `sum by (model_version) (rate(churn_predictions_total{label="churn"}[10m])) / sum by (model_version) (rate(churn_predictions_total[10m]))`
  - Trung vị xác suất: `histogram_quantile(0.5, sum by (le, model_version) (rate(churn_probability_bucket[10m])))`
  - Tỷ lệ lỗi và p95 độ trễ theo `pod`/`instance` (đã có luật cảnh báo toàn cục).
- Tiêu chí chuyển tiếp hoặc lùi: tỷ lệ lỗi không cao hơn bản ổn định, phân phối xác suất không lệch bất thường, độ trễ trong ngưỡng. Đạt thì **duyệt** phiên bản mới (đặt `champion`) và bỏ ghim; không đạt thì giảm trọng số canary về 0.

## 4. A/B testing (thiết kế)

A/B khác canary ở mục tiêu: **đo xem phiên bản nào mang lại kết quả kinh doanh tốt hơn**, không chỉ kiểm tra có chạy ổn hay không. Cần thêm:

1. **Định tuyến cố định theo khách**: băm mã khách hàng để một khách luôn gặp cùng một phiên bản, nếu không kết quả bị lẫn. Request hiện **chưa có mã khách hàng**.
2. **Ghi lại từng dự đoán** (mã khách, phiên bản, xác suất, thời điểm) để sau này nối với việc khách có rời đi hay không. API hiện **không lưu dự đoán**.
3. **Chờ nhãn**: churn chỉ biết sau nhiều tuần, nên kết luận A/B phải dựa trên kết quả đến muộn. Trong lúc chờ chỉ có các số liệu gián tiếp ở mục 3.
4. **Cỡ mẫu và tiêu chí dừng** quyết định trước khi chạy. Với dataset này, chênh lệch giữa các mô hình nằm trong vùng nhiễu (xem kết quả mảng 1), nên cần lượng khách lớn mới phân biệt được.

## 5. Chưa làm

- **Đưa image đã được CI kiểm tra (GHCR) lên cụm.** Deploy tự động đã có (`scripts/deploy-local.sh` qua runner trên Mac) nhưng **build lại** image arm64 trên runner vì image CI là amd64. Muốn deploy đúng image CI cần build đa kiến trúc (hoặc dùng runner arm64 của GitHub) và kéo từ GHCR.
- **Đưa model vào canary tự động.** Duyệt model chỉ đổi alias `champion` cho nhóm stable; canary là việc triển khai tay (`kubectl apply -k k8s/canary`, ghim phiên bản cụ thể). Chưa có luồng "challenger vào canary, đạt tiêu chí thì duyệt".
- Postgres, MinIO, MLflow chưa chạy trong cụm (vẫn ở Docker Compose).
- Mã khách hàng trong request và việc lưu nhật ký dự đoán (cần cho A/B).
- Cách lưu `ADMIN_KEY` an toàn trên K8s (Secret) và bảo vệ các endpoint duyệt khi có nhiều bản sao (hiện chỉ có khóa dùng chung).
- Dashboard Grafana so sánh phiên bản (các truy vấn ở mục 3 mới là gợi ý).
