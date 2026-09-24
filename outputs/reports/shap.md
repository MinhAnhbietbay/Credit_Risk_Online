# SHAP — giải thích model (Task 11)

> **CatBoost fit với scale_pos_weight nên xác suất nó trả về là ĐIỂM RỦI RO CHƯA HIỆU CHỈNH (mức nền ~35% chứ không phải tỉ lệ vỡ nợ thật ~8%); PD dùng cho quyết định lấy từ ensemble.**

Mẫu: **5,000 hồ sơ** lấy phân tầng theo TARGET từ phần CV (không chạm holdout). SHAP **out-of-fold**: mỗi hồ sơ được giải thích bằng đúng 1 trong 5 model fold không train trên hồ sơ đó. Đơn vị: log-odds, cộng được về dự đoán của model.
## Top 20 feature theo |SHAP| trung bình
| # | Feature | \|SHAP\| TB | Ý nghĩa |
|---:|---|---:|---|
| 1 | `EXT_SOURCE_MEAN` | 0.3665 | trung bình 3 điểm tín dụng nguồn ngoài (càng cao càng tốt) |
| 2 | `DAYS_EMPLOYED` | 0.1288 | số ngày kể từ khi vào làm công việc hiện tại (âm = quá khứ) |
| 3 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | 0.1188 | đơn vay trước ở Home Credit: số ngày đến kỳ trả cuối theo lịch ký ban đầu (lệch nhiều so với thực tế = trả sớm/trễ), lớn nhất |
| 4 | `CREDIT_TERM` | 0.1172 | tiền trả mỗi kỳ chia cho tổng khoản vay (nghịch đảo số kỳ ước tính) |
| 5 | `NAME_EDUCATION_TYPE` | 0.1101 | trình độ học vấn cao nhất |
| 6 | `AMT_ANNUITY` | 0.1061 | số tiền phải trả mỗi kỳ |
| 7 | `EXT_SOURCE_1` | 0.0953 | điểm tín dụng từ nguồn ngoài số 1 (0-1, càng cao càng tốt) |
| 8 | `EXT_SOURCE_MAX` | 0.0831 | điểm tín dụng nguồn ngoài cao nhất trong 3 nguồn |
| 9 | `EXT_SOURCE_MIN` | 0.0826 | điểm tín dụng nguồn ngoài thấp nhất trong 3 nguồn |
| 10 | `INST_LATE_RATIO` | 0.0808 | tỉ lệ kỳ trả góp bị trễ hạn trên tổng số kỳ |
| 11 | `DAYS_BIRTH` | 0.0683 | số ngày tính đến ngày sinh (âm; càng âm càng lớn tuổi) |
| 12 | `EXT_SOURCE_3` | 0.0657 | điểm tín dụng từ nguồn ngoài số 3 (0-1, càng cao càng tốt) |
| 13 | `OCCUPATION_TYPE` | 0.0608 | nghề nghiệp |
| 14 | `PREV_DAYS_LAST_DUE_MAX` | 0.0538 | đơn vay trước ở Home Credit: số ngày đến kỳ trả cuối theo hợp đồng (âm = đã qua), lớn nhất |
| 15 | `OWN_CAR_AGE` | 0.0537 | tuổi xe ô tô khách sở hữu (năm) |
| 16 | `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN` | 0.0534 | CIC: dư nợ còn lại, trung bình |
| 17 | `PREV_PRODUCT_COMBINATION_MODE` | 0.0533 | đơn vay trước ở Home Credit: gói sản phẩm vay, giá trị hay gặp nhất |
| 18 | `PREV_REFUSED_RATIO` | 0.0516 | tỉ lệ đơn vay trước bị từ chối trên tổng số đơn |
| 19 | `ORGANIZATION_TYPE` | 0.0494 | loại tổ chức nơi khách làm việc |
| 20 | `PREV_APPROVED_AMT_ANNUITY_SUM` | 0.0479 | đơn vay trước ở Home Credit: số tiền phải trả mỗi kỳ, tính trên đơn đã duyệt, tổng |

Xếp hạng đầy đủ 80 feature: `outputs/reports/shap_global.csv`.

## Giải thích một hồ sơ (local)

Hồ sơ rủi ro cao nhất trong mẫu: **điểm CatBoost 95.1%**, mức nền của model 34.9%. Mức nền cao hơn tỉ lệ vỡ nợ thật (~8%) vì CatBoost fit với `scale_pos_weight` — đây là **điểm xếp hạng, không phải PD**; PD cho quyết định lấy từ ensemble năm yếu tố đẩy mạnh nhất:

1. EXT_SOURCE_MEAN = 0.269 (trung bình 3 điểm tín dụng nguồn ngoài (càng cao càng tốt)) làm rủi ro tăng (SHAP +0.717 log-odds)
2. EXT_SOURCE_MIN = 0.085 (điểm tín dụng nguồn ngoài thấp nhất trong 3 nguồn) làm rủi ro tăng (SHAP +0.284 log-odds)
3. CREDIT_TERM = 0.0486 (tiền trả mỗi kỳ chia cho tổng khoản vay (nghịch đảo số kỳ ước tính)) làm rủi ro tăng (SHAP +0.202 log-odds)
4. PREV_REFUSED_RATIO = 0.556 (tỉ lệ đơn vay trước bị từ chối trên tổng số đơn) làm rủi ro tăng (SHAP +0.200 log-odds)
5. EXT_SOURCE_3 = 0.085 (điểm tín dụng từ nguồn ngoài số 3 (0-1, càng cao càng tốt)) làm rủi ro tăng (SHAP +0.143 log-odds)

Biểu đồ waterfall: `outputs/plots/ph1_shap_waterfall.png`.

## Plot

- `outputs/plots/ph1_shap_beeswarm.png` — beeswarm top-20
- `outputs/plots/ph1_shap_bar.png` — |SHAP| trung bình top-20
- `outputs/plots/ph1_shap_waterfall.png` — waterfall hồ sơ ở trên
- **`EXT_SOURCE_*` chi phối model** (|SHAP| lớn hơn phần còn lại một bậc) nhưng là điểm hộp đen từ nguồn ngoài: không biết nó được tính từ gì, nên không kiểm tra được bản thân nó có thiên lệch hay không.

## Bằng chứng cho cách tính SHAP

SHAP out-of-fold (5 model fold) so với chỉ dùng model fold 0: trùng **16/20** feature ở top-20, tương quan Spearman toàn bảng **0.9464**, lệch |SHAP| trung bình lớn nhất **0.0216** log-odds. Chi tiết: `outputs/evidence/task11-oof-vs-fold0-shap/`.
