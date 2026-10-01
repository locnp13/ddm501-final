# Câu hỏi cần chốt trước khi đi tiếp

Cập nhật: 2026-09-30. Đã đi qua cả 23 câu; hai điểm còn mở: máy chung cụ thể ở Q18, và gán tên người ở Q23. Mỗi câu có **đề xuất mặc định** để nhóm chỉ cần đồng ý hoặc phản biện. Điền cột "Quyết định" khi đã chốt.

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

### Kết quả mảng 1 (2026-09-30)

Đã triển khai Q1-Q4, Q6-Q11, Q13 (code: `features.py`, `business.py`, `registry.py`, `train.py`, `promote.py`). Run thật trên Telco: model `churn-model` v2 (XGBoost đã tune, không dùng feature mới, bỏ `gender`/`SeniorCitizen`) qua gate, được duyệt lên `champion`; API nạp `models:/churn-model@champion` và dự đoán được.

CV PR-AUC (5-fold, OOF gộp) của các cấu hình chính:

| Cấu hình | CV PR-AUC |
|---|---|
| Dummy (mốc theo tỷ lệ churn) | 0.265 |
| Random Forest (tốt nhất) | 0.656 |
| Logistic Regression + feature mới, bỏ cột nhạy cảm | 0.659 |
| XGBoost mặc định | 0.658 |
| XGBoost đã tune, không feature mới, bỏ cột nhạy cảm (được chọn) | 0.666 |

Kết quả trên test (1.409 dòng): PR-AUC 0.663, ROC-AUC 0.848; Brier 0.1355 sau hiệu chuẩn (0.1604 trước); lợi nhuận thực tế tối đa 30.614 khi liên hệ 35% khách có điểm cao nhất (recall 0.74, precision 0.56), theo các giả định ở Q1.

Những điều rút ra, cần nói thật trong báo cáo:

- **Mọi mô hình khác Dummy đều nằm trong khoảng 0.65-0.67**: dataset này gần như đã "bão hòa", chênh lệch giữa Logistic Regression, Random Forest và XGBoost nhỏ hơn nhiễu. XGBoost được chọn chỉ nhờ hơn khoảng 0.004-0.008 trên CV.
- **Feature mới (Q6) không cải thiện rõ**: hầu hết cặp có/không có feature chênh dưới 0.002 (cây quyết định đã tự học được các quan hệ này). Kết quả trung thực để đưa vào báo cáo, không phải lỗi.
- **Cột nhạy cảm (Q7)**: bỏ `gender`/`SeniorCitizen` gần như không mất PR-AUC (quy tắc chọn dùng dung sai 0.005). `gender` gần như không ảnh hưởng chênh lệch chọn khách (0.002); `SeniorCitizen` vẫn chênh 0.09-0.17 ở top 10% trong CV (0.11 trên test) vì nhóm này có tỷ lệ churn thực tế cao hơn, và bỏ cột không loại được vì còn các biến proxy. Cần xử lý tiếp ở mục fairness (mảng 4).
- **`scale_pos_weight` (Q4)**: với hiệu chuẩn isotonic, bỏ nó không đổi chất lượng (CV cùng loại 0.6702 với, 0.6711 không). Có thể bỏ ở bước sau để đơn giản hóa.
- **API vẫn gắn nhãn churn ở ngưỡng 0.5** (`app.py`, ngoài phạm vi mảng 1): sau hiệu chuẩn, xác suất là xác suất thật nên cần xem lại; thuộc mảng 2.

## 2. Câu hỏi về bài toán và metric

| # | Câu hỏi | Vì sao quan trọng | Đề xuất mặc định | Quyết định |
|---|---|---|---|---|
| Q1 | Chi phí giữ chân một khách (ưu đãi) so với giá trị một khách rời đi là bao nhiêu? | Quyết định ngưỡng phân loại. Ngưỡng 0.5 hiện chỉ là mặc định, chưa gắn với chi phí nào | Giả định chi phí và giá trị, ghi rõ trong tài liệu, chọn ngưỡng theo lợi nhuận kỳ vọng | **Đã chốt (cách A)**: giá trị mất khách = `MonthlyCharges` x 12; chi phí ưu đãi = 10% doanh thu năm của khách; tỷ lệ giữ chân thành công = 30%. Chọn ngưỡng theo lợi nhuận kỳ vọng. Ba tham số này là giả định, phải nêu rõ trong tài liệu. |
| Q2 | Ngân sách chăm sóc chỉ đủ cho top-k% khách hàng? | Nếu có, metric đúng là Recall@k hoặc lift, không phải F1 ở 0.5 | Thêm Recall@10% và lift@10% cùng PR-AUC | **Đã chốt (cách B)**: quét k từ 5% đến 50%, báo cáo Recall@k và lợi nhuận kỳ vọng theo k (dùng mô hình lợi nhuận của Q1), chọn k tối ưu; vẽ biểu đồ lợi nhuận theo k. PR-AUC vẫn là metric chính để chọn mô hình. |
| Q3 | Baseline để so sánh là gì? | Không có baseline thì không chứng minh được mô hình tốt hơn | Dummy (tần suất churn ~0.27 làm PR-AUC nền), Logistic Regression, Random Forest, rồi XGBoost | **Đã chốt (cách A)**: 4 mô hình Dummy, Logistic Regression, Random Forest, XGBoost. Mỗi mô hình một run MLflow, cùng preprocessing và cùng CV; chỉ mô hình có CV PR-AUC cao nhất đi tiếp vào huấn luyện cuối và gate. MLP/LightGBM để sau nếu còn thời gian. |
| Q4 | Có cần hiệu chuẩn xác suất (calibration) không? | `scale_pos_weight` làm xác suất bị đẩy lên (ví dụ mẫu ra 0.80). Nếu đưa "% nguy cơ" cho người dùng thì số này sai lệch | Đo Brier score và đường calibration; hiệu chuẩn (isotonic) nếu dùng xác suất như rủi ro | **Đã chốt (cách A)**: giữ `scale_pos_weight`, hiệu chuẩn isotonic bằng `CalibratedClassifierCV` (5-fold), đo Brier score và vẽ đường calibration trước/sau; model đăng ký trả xác suất đã hiệu chuẩn (cần cho mô hình lợi nhuận Q1/Q2). Thử thêm biến thể bỏ `scale_pos_weight` như một experiment phụ. |
| Q5 | Dataset chính là Telco hay E-Commerce Churn? | Telco không có cột thời gian nên chia train/test ngẫu nhiên, không kiểm tra được drift theo thời gian | Giữ Telco, nêu rõ hạn chế này trong tài liệu | **Đã chốt (cách A)**: giữ Telco (nằm trong danh sách dataset gợi ý của đề). Ghi rõ hạn chế: ngành viễn thông thay vì e-commerce, không có cột thời gian nên chia train/test ngẫu nhiên và không kiểm tra được drift theo thời gian. |

## 3. Câu hỏi về dữ liệu và feature

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q6 | Feature engineering hiện gần như không có (chỉ impute và one-hot). Có làm thêm không (tenure bucket, số dịch vụ đang dùng, chi phí/tháng so với trung bình)? Đề nêu feature engineering là thách thức chính | Thêm 3-5 feature có lý do nghiệp vụ, so sánh trước/sau bằng MLflow | **Đã chốt (cách A)**: thêm 3-5 feature (`tenure_bucket`, `num_services`, `avg_charge_per_month`, `autopay`, có thể `has_internet_no_support`). Feature nằm **trong pipeline** (transformer trước `ColumnTransformer`) để API áp dụng cùng phép biến đổi. Ghi hai nhóm experiment lên MLflow (có / không có feature mới) cùng CV. |
| Q7 | Cột nhạy cảm `gender`, `SeniorCitizen` đang là feature của model. Giữ, bỏ, hay chỉ dùng để kiểm định fairness? (Spec ghi "Ask First") | Giữ trong dữ liệu để đo fairness, huấn luyện thử cả hai và so sánh; quyết định dựa trên kết quả | **Đã chốt (cách B)**: chạy hai experiment (có / không có `gender`, `SeniorCitizen` làm feature), so PR-AUC và các chỉ số fairness rồi chọn. Hai cột luôn được giữ trong dữ liệu để chia nhóm khi đo fairness. Lưu ý: bỏ cột không loại được proxy (`Partner`, `Dependents`, loại hợp đồng), và việc dùng `SeniorCitizen` để quyết định ưu đãi phải được nêu trong phần ethics dù chọn hướng nào. |
| Q8 | Có ghi lại phiên bản dữ liệu và git commit vào mỗi run MLflow không? Hiện run chưa có git SHA (container không có git) | Log hash của `dvc.lock` và commit vào tag của run để tái lập được | **Đã chốt (cách A)**: compose truyền `GIT_COMMIT` và cờ `GIT_DIRTY` vào container (CI dùng `github.sha`); `train.py` đọc thêm md5 của `data/processed/churn.csv` trong `dvc.lock` và log thành tag của run, cảnh báo khi thiếu `GIT_COMMIT`. Không cài git vào image. |

## 4. Câu hỏi về huấn luyện và quy trình model

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q9 | Ai được phép đưa model lên Production: tự động khi qua gate, hay cần người duyệt? | Tự động lên `Staging` khi qua gate, người duyệt chuyển `Production` (hoặc so với model hiện tại: chỉ thăng hạng khi PR-AUC không kém) | **Đã chốt (cách A+B)**: luồng `train -> gate (so với champion hiện tại) -> Staging/challenger -> một thành viên duyệt và chạy lệnh thăng hạng -> Production`. Chi tiết ngưỡng ở Q10, cơ chế alias ở Q11. **Cập nhật 2026-10-01**: nhóm yêu cầu không tự động duyệt và duyệt trên UI: tab Mô hình hiện bản `challenger` cạnh `champion` kèm nút Duyệt; `POST /v1/model/promote` cần khóa `ADMIN_KEY` (trong `.env`, rỗng thì tắt), chỉ nhận đúng bản đang là challenger, nạp thử model trước khi đổi alias, ghi tag `approved_at` và phục vụ ngay không cần restart. Quay về bản cũ: `POST /v1/model/rollback` và nút Khôi phục (xem `docs/rollout-strategies.md`, cũng mô tả hướng canary/A-B trên Kubernetes). v3 là bản do nhóm duyệt lúc 2026-10-01 04:15 UTC. |
| Q10 | Ngưỡng gate 0.50 có ý nghĩa gì khi kết quả là 0.66? | Đặt ngưỡng = baseline tốt nhất từ Q3 cộng một biên, và thêm điều kiện không tệ hơn model hiện tại | **Đã chốt (cách A)**: gate gồm 3 điều kiện: (1) PR-AUC >= 0.50 (sàn tuyệt đối); (2) PR-AUC >= PR-AUC của Logistic Regression + 0.02; (3) PR-AUC >= champion hiện tại - 0.005 (bỏ qua ở lần chạy đầu khi chưa có champion). Các biên 0.02 và 0.005 là đề xuất chưa đo đạc, đặt trong `params.yaml`, chỉnh lại theo độ lệch chuẩn CV PR-AUC khi có số liệu. **Sửa ngày 2026-09-30 sau khi đo (biên 0.02 -> 0.0)**: XGBoost trừ LogReg trên 25 fold CV có trung bình +0.0035 (độ lệch chuẩn 0.007, XGBoost thắng 19/25); trên test là +0.0147 với khoảng tin cậy bootstrap 95% (-0.004; +0.034), tức nằm trong nhiễu. Biên 0.02 sẽ từ chối mọi mô hình phức tạp, nên đặt 0.0 (chỉ yêu cầu không tệ hơn baseline). Điều kiện baseline bỏ qua khi mô hình thắng là `logreg` hoặc `dummy` (không so với chính nó). |
| Q11 | Model Registry stage (`Production`) đã bị MLflow đánh dấu deprecated. Đổi sang alias (`champion`) hay giữ? | Đổi sang alias ngay bây giờ để khỏi phải sửa API sau (`models:/churn-model@champion`) | **Đã chốt (cách A)**: dùng alias `challenger` (qua gate) và `champion` (đã duyệt). Sửa 3 chỗ: `train.py` (`set_registered_model_alias`), `MODEL_URI` mặc định trong `src/serving/app.py` và `docker-compose.yml` (`models:/churn-model@champion`), và thêm script duyệt đặt alias `champion`. Việc sửa `app.py` là ngoại lệ so với ràng buộc "Never" của spec đợt trước. |
| Q12 | Báo cáo Evidently đang so train với test (chia ngẫu nhiên nên gần như không có drift, ít giá trị). Giữ làm gì? | Giữ như kiểm tra nhanh, chuyển trọng tâm sang so dữ liệu serving với tập train (xem Q16) | **Đã chốt (cách A)**: giữ báo cáo như sanity check xác nhận chia train/test không bị lệch. Không gọi là drift detection trong tài liệu hay slide vì hai tập chia ngẫu nhiên từ cùng nguồn. Drift thật (serving vs train) làm ở Q16/Q17. |
| Q13 | Tập test dùng cho cả báo cáo và gate. Có cần thêm tập validation riêng không? | Không cần thêm vì tuning dùng CV trên tập train; chỉ ghi rõ test chỉ đánh giá một lần | **Đã chốt (cách A)**: giữ hai tập (train 80% / test 20%). Quy tắc: chọn mô hình, feature (Q6), cột nhạy cảm (Q7) và siêu tham số **chỉ bằng CV PR-AUC** trên train; test dùng đúng một lần cho mô hình đã chọn (báo cáo và gate). Ghi quy tắc này vào tài liệu. |

## 5. Câu hỏi về serving và giám sát

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q14 | API hiện nhận `dict[str, Any]` không kiểm tra. Dùng schema Pydantic (sinh từ schema Pandera) để Swagger có ví dụ và trả 422 khi sai? | Có, cộng thêm `/v1/predict/batch` và trả về kèm `model_version` | **Đã chốt (cách A); đã làm 2026-10-01 trừ `/v1/predict/batch`**: model Pydantic `CustomerFeatures` viết tay (kiểu, dải giá trị; cột phân loại khai báo `str` và kiểm tra với tập giá trị đã biết để sinh `warnings`, không dùng `Literal` vì Q15 cho phép category lạ) với ví dụ OpenAPI, trả `422` khi sai; thêm `/v1/predict/batch` (giới hạn kích thước) và `model_version` trong phản hồi. Một test tự động so schema Pydantic với `SCHEMA` Pandera để bắt lệch (không thêm Pandera vào image API). API chỉ nhận cột thô, feature mới (Q6) nằm trong pipeline. |
| Q15 | Giá trị ngoài dải (category lạ, `TotalCharges` rỗng) thì trả lỗi hay vẫn dự đoán? | Category lạ vẫn dự đoán (encoder đã bỏ qua), thiếu trường bắt buộc thì 422 | **Đã chốt (cách A); đã làm 2026-10-01**: `422` khi thiếu trường, sai kiểu hoặc số ngoài dải (`tenure`, `MonthlyCharges` < 0). `TotalCharges` cho phép `null` (pipeline impute như lúc train). Category lạ vẫn dự đoán, phản hồi kèm `warnings` và tăng metric Prometheus `churn_unknown_category_total` (cũng là tín hiệu drift cho Q16). |
| Q16 | Churn có nhãn muộn nên không đo được độ chính xác theo thời gian thực. Giám sát chỉ dựa trên drift dữ liệu và phân phối dự đoán. Chấp nhận? Khi nào retrain? | Chấp nhận, ghi rõ hạn chế; trigger retrain khi PSI > 0.2 hoặc theo lịch tháng | **Đã chốt (cách A)**: giám sát data drift (PSI cho các feature chính so với thống kê lưu lúc train, đi kèm artifact của model) và prediction drift (phân phối xác suất, tỷ lệ dự đoán churn), cộng tín hiệu chất lượng dữ liệu từ Q15. Cảnh báo Prometheus khi PSI > 0.2 (0.1-0.2 là cảnh báo nhẹ). Retrain khi vượt ngưỡng hoặc mỗi tháng. Ghi rõ trong tài liệu: giám sát **không đo được độ chính xác thật** vì nhãn churn đến muộn, chỉ báo hiệu dữ liệu đã đổi. |
| Q17 | Dữ liệu "production" để kiểm tra drift lấy từ đâu khi đồ án không có khách thật? | Viết script mô phỏng traffic (lấy mẫu từ tập test, có kịch bản làm lệch phân phối) để demo alert | **Đã chốt (cách A)**: script trong `simulation/` gửi khách lấy mẫu từ tập test theo hai chế độ `normal` (PSI thấp, không alert) và `drift` (bóp méo có kiểm soát: nhiều hợp đồng tháng-qua-tháng hơn, `tenure` ngắn hơn, `MonthlyCharges` cao hơn; PSI vượt 0.2, alert kích hoạt), tốc độ chỉnh được. Có thể thêm chế độ gửi dữ liệu sai để tăng tỷ lệ 422. Ghi rõ trong README và tài liệu đây là dữ liệu mô phỏng, kèm kịch bản làm lệch. |

## 6. Câu hỏi về hạ tầng, bảo mật, tổ chức

| # | Câu hỏi | Đề xuất mặc định | Quyết định |
|---|---|---|---|
| Q18 | Cả nhóm dùng chung MinIO/MLflow ở đâu? Hiện chạy cục bộ trên máy từng người nên `dvc pull` của người khác sẽ không thấy dữ liệu | Một VM chung (hoặc máy của một thành viên) chạy stack, mọi người trỏ `MLFLOW_TRACKING_URI` và endpoint DVC tới đó | **Đã chốt (cách A)**: một máy chung chạy stack Compose; mọi người trỏ `MLFLOW_TRACKING_URI` và endpoint DVC (`.dvc/config.local`) tới máy đó. **Còn mở**: máy cụ thể chưa xác định (VM đám mây hay máy luôn bật của một thành viên), cần nhóm chỉ định người phụ trách. Không mở cổng 9000/5001 công khai, dùng VPN hoặc tunnel SSH, và đổi mật khẩu mặc định (Q20). |
| Q19 | CI/CD deploy đi đâu? Hiện chỉ đẩy image API lên GHCR | VM chung ở Q18, deploy bằng SSH + `docker compose pull && up -d` | **Đã chốt (self-hosted runner)**: cài GitHub Actions runner trên Mac; runner tự kết nối ra GitHub nên không cần mở cổng hay SSH. Job `deploy` (`runs-on: self-hosted`) chạy sau `build`, chỉ trên `push` vào `main`: `docker compose pull api && docker compose up -d api`, rồi kiểm tra `/health`. Chỉ restart service `api`, không đụng MLflow/MinIO/Postgres; compose trên Mac phải dùng image từ GHCR thay vì build. Máy tắt hoặc ngủ thì không deploy và API không phục vụ, cần ghi vào tài liệu vận hành. **Rủi ro chưa xử lý**: repo đang PUBLIC, mà self-hosted runner trên repo public có thể chạy code từ PR của người lạ ngay trên Mac; phải chuyển repo sang private, hoặc giữ public kèm cấu hình chặn (job deploy không chạy trên `pull_request`, bật yêu cầu phê duyệt workflow từ cộng tác viên bên ngoài). **Đã chốt: giữ public + cấu hình chặn**: (1) job `deploy` chỉ có điều kiện `github.event_name == 'push' && github.ref == 'refs/heads/main'`, các job chạy trên PR (`lint-test`, `build`) giữ `runs-on: ubuntu-latest`, không bao giờ chạy trên runner tự host; (2) Settings > Actions > General: đặt "Fork pull request workflows from outside collaborators" thành "Require approval for all outside collaborators"; (3) bảo vệ nhánh `main` (bắt buộc PR và CI xanh trước khi merge); (4) runner chạy dưới tài khoản người dùng thường, không dùng quyền root. Rủi ro còn lại: cấu hình sai hoặc một thành viên duyệt nhầm PR độc hại thì code vẫn chạy trên Mac. Nếu muốn loại bỏ hẳn, chuyển repo sang private. Cách tunnel + SSH (Tailscale) là phương án dự phòng, nếu dùng thêm thì cũng giải Q18 cho cả nhóm. |
| Q20 | Thông tin đăng nhập mặc định (`minioadmin`, Grafana `admin/admin`, Postgres `mlflow/mlflow`) có chấp nhận cho bản nộp không? | Chuyển sang biến môi trường trong `.env` (không commit), ghi rõ trong hướng dẫn vận hành | **Đã chốt (cách A)**: compose đọc `MINIO_ROOT_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`, `POSTGRES_PASSWORD` từ `.env` (đã gitignore), dùng `${VAR:?thiếu biến}` để từ chối chạy khi thiếu; commit `.env.example` chỉ có placeholder; README thêm lệnh `cp .env.example .env`. Các giá trị mặc định cũ đã nằm trong lịch sử git nên máy chung phải dùng mật khẩu mới. |
| Q21 | Image MinIO chính thức không còn phát hành; đang dùng bản `bitnamilegacy` đóng băng. Chấp nhận, hay đổi sang thay thế (SeaweedFS, Garage)? | Chấp nhận, nêu trong phần tech-stack justification | **Đã chốt (cách A)**: giữ `bitnamilegacy/minio:2024.12.18`, nêu trong tech-stack justification và README rằng bản này đóng băng (không nhận bản vá), không dùng cho production. Lưu bản sao để khỏi mất khi tag bị gỡ: đề xuất `docker save` ra tệp hoặc GHCR **private** (MinIO dùng AGPLv3, nên xem điều khoản trước khi đẩy bản sao lên registry công khai). |
| Q22 | Cách đạt coverage > 80% khi `train.py` cần MLflow? | Test `tune`/`evaluate`/`build_pipeline` trên dữ liệu nhỏ, chạy `main` với MLflow trỏ vào thư mục tạm (SQLite) | **Đã chốt (cách A)**: ba tầng test. (1) Unit test hàm thuần (`passes_gate`, `evaluate`, `build_pipeline`, PSI, feature mới). (2) Test "smoke" chạy `main()` với MLflow SQLite trong thư mục tạm, khoảng 300 dòng dữ liệu tổng hợp, `n_trials=2`, kiểm tra luồng gate và đặt alias; tách `main()` thành các hàm nhỏ để dễ test. Test này bắt được lỗi kiểu xgboost/sklearn mà CI từng bỏ sót. (3) Test API bằng `TestClient` với model giả (200, 422, 503, batch, metric Prometheus). Bật `--cov-fail-under=80` trong CI sau khi đạt, không loại `train.py` khỏi phạm vi đo. |
| Q23 | Ai phụ trách mảng nào (ML pipeline, serving/CI, monitoring/test, responsible AI/docs)? | Điền vào cột "Phụ trách" trong README | **Ghi đề xuất, chưa gán tên**: 4 mảng: (1) ML pipeline (Q1-Q4, Q6, Q8-Q11; 15%); (2) Serving và deploy (Q14-Q15, Q19-Q21; 15%); (3) Monitoring và test (Q16-Q17, Q22, Grafana; 10% + 15%); (4) Responsible AI và tài liệu (SHAP, LIME, fairness Q7, privacy, ethics, kiến trúc, README; 10% + 10%). Mảng 4 (fairness) cần kết quả mảng 1 nên làm sau hoặc phối hợp sớm; mảng 3 nặng nhất và phụ thuộc code của mảng 1, 2 nên dễ dồn về cuối. Mỗi người sở hữu thư mục riêng, làm trên branch riêng và mở PR. Nhóm tự gán tên và cập nhật cột "Phụ trách" trong README. |

## 7. Thứ tự đề xuất

1. Chốt Q1-Q5, Q7 (định nghĩa bài toán). Đây là 10% điểm "problem definition" và chi phối mọi thứ sau.
2. Thêm baseline và feature (Q3, Q6), ghi nhiều experiment lên MLflow.
3. Schema API và test (Q14, Q22) để kéo coverage.
4. Giám sát và drift (Q16, Q17), dashboard Grafana.
5. Responsible AI (SHAP, LIME, fairness) sau khi Q7 đã rõ.
6. Hạ tầng dùng chung và deploy (Q18, Q19).
