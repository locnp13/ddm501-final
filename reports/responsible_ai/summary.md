# Báo cáo Responsible AI (tự sinh)

> Bản trong repo được tạo từ một lần huấn luyện tái lập trên cùng code, dữ liệu và `params.yaml` (PR-AUC test 0,661; bản `champion` v3 của nhóm là 0,663), nên "phiên bản 1" bên dưới là phiên bản trong kho MLflow tạm, không phải v3. Chạy lại `docker compose run --rm trainer python -m src.responsible.report --log-to-run` để cập nhật cho champion thật (ghi đè tệp này).

Tạo bởi `python -m src.responsible.report` cho model `models:/churn-model@champion` (phiên bản 1). Giải thích ý nghĩa: `docs/responsible-ai.md`.

Tập test: 1409 khách. Quyết định đánh giá: liên hệ 35.0% khách có điểm cao nhất (k tối ưu lợi nhuận của lần huấn luyện).

## 1. Giải thích mô hình

SHAP (permutation, 200 khách, nền 50 khách huấn luyện; xác suất trung bình nền 0.293; sai số cộng tính lớn nhất 0.0e+00).

| Đầu vào | Trung bình \|SHAP\| |
|---|---|
| `Contract` | 0.0875 |
| `tenure` | 0.0663 |
| `InternetService` | 0.0481 |
| `MonthlyCharges` | 0.0328 |
| `OnlineSecurity` | 0.0311 |
| `TotalCharges` | 0.0262 |
| `PaymentMethod` | 0.0245 |
| `TechSupport` | 0.0243 |
| `PaperlessBilling` | 0.0192 |
| `MultipleLines` | 0.0140 |

Ảnh hưởng trung bình theo giá trị (dương: đẩy về phía rời bỏ):

- `Contract`: Month-to-month +0.075, One year -0.083, Two year -0.127
- `InternetService`: Fiber optic +0.049, No -0.041, DSL -0.051
- `PaymentMethod`: Electronic check +0.033, Credit card (automatic) -0.020, Bank transfer (automatic) -0.020, Mailed check -0.022
- `TechSupport`: No +0.024, No internet service -0.023, Yes -0.026

LIME so với SHAP: trung bình 72.0% trong 5 đầu vào LIME xếp cao nhất cũng nằm trong top 5 của SHAP (cùng khách).

## 2. Công bằng của quyết định liên hệ

| Nhóm | Số khách | Tỷ lệ churn thật | Xác suất TB | Tỷ lệ được liên hệ | TPR | FPR | Precision |
|---|---|---|---|---|---|---|---|
| SeniorCitizen=0 | 1187 | 23.2% | 24.2% | 30.3% | 70.7% | 18.1% | 54.2% |
| SeniorCitizen=1 | 222 | 44.1% | 39.9% | 59.9% | 84.7% | 40.3% | 62.4% |
| gender=Female | 687 | 28.1% | 26.9% | 35.2% | 73.1% | 20.4% | 58.3% |
| gender=Male | 722 | 25.1% | 26.6% | 34.8% | 75.7% | 21.1% | 54.6% |

- `SeniorCitizen`: chênh tỷ lệ được liên hệ 29.6%, chênh TPR 14.0%, tỷ số disparate impact 0.51. Đoán `SeniorCitizen` từ các cột còn lại: ROC-AUC 0.79.
- `gender`: chênh tỷ lệ được liên hệ 0.5%, chênh TPR 2.6%. Đoán `gender` từ các cột còn lại: ROC-AUC 0.47.

## 3. Giảm thiên lệch: ngưỡng theo nhóm (equal opportunity) cho `SeniorCitizen`

Ngưỡng học trên một nửa tập test, đánh giá trên nửa còn lại, lặp 50 lần (trung bình ± độ lệch chuẩn), cùng ngân sách liên hệ.

| Quyết định | Chênh TPR | Chênh tỷ lệ được liên hệ | Tỷ lệ liên hệ | Lợi nhuận (nửa tập test) |
|---|---|---|---|---|
| top-k hiện tại | 12.6% ± 4.7% | 29.1% | 35.0% | 15,630 ± 1,047 |
| ngưỡng theo nhóm | 9.6% ± 6.5% | 18.2% | 34.8% | 14,991 ± 1,098 |

Hình: `shap_importance.png`, `local_explanations.png`, `fairness_senior.png`.
