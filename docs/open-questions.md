# Câu hỏi cần chốt trước khi đi tiếp

Cập nhật: 2026-09-30. Mỗi câu có **đề xuất mặc định** để nhóm chỉ cần đồng ý hoặc phản biện. Điền cột "Quyết định" khi đã chốt.

## 1. Trạng thái code huấn luyện

**Kết luận: pipeline chạy được end-to-end (đã kiểm chứng), nhưng mới đáp ứng phần "chạy được một lần đúng", chưa đủ yêu cầu ML pipeline của đề.**

Đã kiểm chứng ngày 2026-09-30 bằng `docker compose run --rm trainer dvc repro`:

| Tiêu chí (spec) | Kết quả |
|---|---|
| Có run MLflow với params, metrics, báo cáo Evidently, model `churn-model` ở Production | Đạt (v1) |
| Sau khi restart `api`, `/health` báo `model_loaded: true` và `/v1/predict` trả xác suất | Đạt |
| Chạy `dvc repro` lần hai thì bỏ qua các stage | Đạt |
| `pytest` | 8 test qua |

Kết quả trên tập test (20% dữ liệu, 1.409 dòng): PR-AUC 0.664, ROC-AUC 0.847, precision 0.528, recall 0.794, F1 0.634 (ngưỡng 0.5).

Hai lỗi đã sửa khi chạy thật, mà CI không bắt được vì test không gọi tới bước fit:

- `xgboost==2.1.3` không tương thích `scikit-learn==1.6.0` (CV trả `nan` cho mọi trial). Đã nâng lên `2.1.4` ở cả image train và API.
- Phụ thuộc gián tiếp `multimethod`, `pathspec` phải ghim phiên bản.

Còn thiếu so với đề:

- Chỉ có **một** họ mô hình (XGBoost). Đề yêu cầu "multiple experiments"; chưa có baseline để chứng minh 0.664 là tốt.
- Chưa có SHAP/LIME, fairness, drift lúc serving (spec đã loại khỏi phạm vi đợt này).
- Coverage khoảng 41%, `train.py` gần như chưa được test.
- Gate hiện chỉ so với ngưỡng cố định 0.50, không so với model đang Production.

## 2. Câu hỏi về bài toán và metric

| # | Câu hỏi | Vì sao quan trọng | Đề xuất mặc định | Quyết định |
|---|---|---|---|---|
| Q1 | Chi phí giữ chân một khách (ưu đãi) so với giá trị một khách rời đi là bao nhiêu? | Quyết định ngưỡng phân loại. Ngưỡng 0.5 hiện chỉ là mặc định, chưa gắn với chi phí nào | Giả định chi phí và giá trị, ghi rõ trong tài liệu, chọn ngưỡng theo lợi nhuận kỳ vọng | |
| Q2 | Ngân sách chăm sóc chỉ đủ cho top-k% khách hàng? | Nếu có, metric đúng là Recall@k hoặc lift, không phải F1 ở 0.5 | Thêm Recall@10% và lift@10% cùng PR-AUC | |
| Q3 | Baseline để so sánh là gì? | Không có baseline thì không chứng minh được mô hình tốt hơn | Dummy (tần suất churn ~0.27 làm PR-AUC nền), Logistic Regression, Random Forest, rồi XGBoost | |
| Q4 | Có cần hiệu chuẩn xác suất (calibration) không? | `scale_pos_weight` làm xác suất bị đẩy lên (ví dụ mẫu ra 0.80). Nếu đưa "% nguy cơ" cho người dùng thì số này sai lệch | Đo Brier score và đường calibration; hiệu chuẩn (isotonic) nếu dùng xác suất như rủi ro | |
| Q5 | Dataset chính là Telco hay E-Commerce Churn? | Telco không có cột thời gian nên chia train/test ngẫu nhiên, không kiểm tra được drift theo thời gian | Giữ Telco, nêu rõ hạn chế này trong tài liệu | |

## 3. Câu hỏi về dữ liệu và feature

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q6 | Feature engineering hiện gần như không có (chỉ impute và one-hot). Có làm thêm không (tenure bucket, số dịch vụ đang dùng, chi phí/tháng so với trung bình)? Đề nêu feature engineering là thách thức chính | Thêm 3-5 feature có lý do nghiệp vụ, so sánh trước/sau bằng MLflow | |
| Q7 | Cột nhạy cảm `gender`, `SeniorCitizen` đang là feature của model. Giữ, bỏ, hay chỉ dùng để kiểm định fairness? (Spec ghi "Ask First") | Giữ trong dữ liệu để đo fairness, huấn luyện thử cả hai và so sánh; quyết định dựa trên kết quả | |
| Q8 | Có ghi lại phiên bản dữ liệu và git commit vào mỗi run MLflow không? Hiện run chưa có git SHA (container không có git) | Log hash của `dvc.lock` và commit vào tag của run để tái lập được | |

## 4. Câu hỏi về huấn luyện và quy trình model

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q9 | Ai được phép đưa model lên Production: tự động khi qua gate, hay cần người duyệt? | Tự động lên `Staging` khi qua gate, người duyệt chuyển `Production` (hoặc so với model hiện tại: chỉ thăng hạng khi PR-AUC không kém) | |
| Q10 | Ngưỡng gate 0.50 có ý nghĩa gì khi kết quả là 0.66? | Đặt ngưỡng = baseline tốt nhất từ Q3 cộng một biên, và thêm điều kiện không tệ hơn model hiện tại | |
| Q11 | Model Registry stage (`Production`) đã bị MLflow đánh dấu deprecated. Đổi sang alias (`champion`) hay giữ? | Đổi sang alias ngay bây giờ để khỏi phải sửa API sau (`models:/churn-model@champion`) | |
| Q12 | Báo cáo Evidently đang so train với test (chia ngẫu nhiên nên gần như không có drift, ít giá trị). Giữ làm gì? | Giữ như kiểm tra nhanh, chuyển trọng tâm sang so dữ liệu serving với tập train (xem Q16) | |
| Q13 | Tập test dùng cho cả báo cáo và gate. Có cần thêm tập validation riêng không? | Không cần thêm vì tuning dùng CV trên tập train; chỉ ghi rõ test chỉ đánh giá một lần | |

## 5. Câu hỏi về serving và giám sát

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q14 | API hiện nhận `dict[str, Any]` không kiểm tra. Dùng schema Pydantic (sinh từ schema Pandera) để Swagger có ví dụ và trả 422 khi sai? | Có, cộng thêm `/v1/predict/batch` và trả về kèm `model_version` | |
| Q15 | Giá trị ngoài dải (category lạ, `TotalCharges` rỗng) thì trả lỗi hay vẫn dự đoán? | Category lạ vẫn dự đoán (encoder đã bỏ qua), thiếu trường bắt buộc thì 422 | |
| Q16 | Churn có nhãn muộn nên không đo được độ chính xác theo thời gian thực. Giám sát chỉ dựa trên drift dữ liệu và phân phối dự đoán. Chấp nhận? Khi nào retrain? | Chấp nhận, ghi rõ hạn chế; trigger retrain khi PSI > 0.2 hoặc theo lịch tháng | |
| Q17 | Dữ liệu "production" để kiểm tra drift lấy từ đâu khi đồ án không có khách thật? | Viết script mô phỏng traffic (lấy mẫu từ tập test, có kịch bản làm lệch phân phối) để demo alert | |

## 6. Câu hỏi về hạ tầng, bảo mật, tổ chức

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q18 | Cả nhóm dùng chung MinIO/MLflow ở đâu? Hiện chạy cục bộ trên máy từng người nên `dvc pull` của người khác sẽ không thấy dữ liệu | Một VM chung (hoặc máy của một thành viên) chạy stack, mọi người trỏ `MLFLOW_TRACKING_URI` và endpoint DVC tới đó | |
| Q19 | CI/CD deploy đi đâu? Hiện chỉ đẩy image API lên GHCR | VM chung ở Q18, deploy bằng SSH + `docker compose pull && up -d` | |
| Q20 | Thông tin đăng nhập mặc định (`minioadmin`, Grafana `admin/admin`, Postgres `mlflow/mlflow`) có chấp nhận cho bản nộp không? | Chuyển sang biến môi trường trong `.env` (không commit), ghi rõ trong hướng dẫn vận hành | |
| Q21 | Image MinIO chính thức không còn phát hành; đang dùng bản `bitnamilegacy` đóng băng. Chấp nhận, hay đổi sang thay thế (SeaweedFS, Garage)? | Chấp nhận, nêu trong phần tech-stack justification | |
| Q22 | Cách đạt coverage > 80% khi `train.py` cần MLflow? | Test `tune`/`evaluate`/`build_pipeline` trên dữ liệu nhỏ, chạy `main` với MLflow trỏ vào thư mục tạm (SQLite) | |
| Q23 | Ai phụ trách mảng nào (ML pipeline, serving/CI, monitoring/test, responsible AI/docs)? | Điền vào cột "Phụ trách" trong README | |

## 7. Thứ tự đề xuất

1. Chốt Q1-Q5, Q7 (định nghĩa bài toán). Đây là 10% điểm "problem definition" và chi phối mọi thứ sau.
2. Thêm baseline và feature (Q3, Q6), ghi nhiều experiment lên MLflow.
3. Schema API và test (Q14, Q22) để kéo coverage.
4. Giám sát và drift (Q16, Q17), dashboard Grafana.
5. Responsible AI (SHAP, LIME, fairness) sau khi Q7 đã rõ.
6. Hạ tầng dùng chung và deploy (Q18, Q19).
