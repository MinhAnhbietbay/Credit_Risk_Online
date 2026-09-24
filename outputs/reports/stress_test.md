# Stress-test: chi phí trả nợ và hành vi trả nợ

> **Phân tích độ nhạy, KHÔNG phải dự báo: hệ số kịch bản là giả định về mức gánh nặng trả nợ, không có quan hệ nhân quả nào giữa biến vĩ mô và vỡ nợ được kiểm chứng trên dữ liệu này.**

Gánh nặng trả nợ đo bằng `AMT_ANNUITY / AMT_CREDIT (= feature CREDIT_TERM)`. Phân vị đo trên **245,998 hồ sơ** của phần CV: p50 = 0.05000, p75 = 0.06391, p90 = 0.09498.

Kênh **hành vi** sốc các cột: `INST_LATE_RATIO`, `INST_DPD_MEAN`, `INST_DPD_STD`, `POS_SK_DPD_DEF_MEAN`, `CC_UTILIZATION_MEAN`, `CC_UTILIZATION_MAX` — mỗi hồ sơ nhích lên ngần ấy điểm phần trăm trong phân phối của chính danh mục (`quantile_shift`), vì các cột này có phân vị 25 bằng 0 nên nhân hệ số vô tác dụng.

| Kịch bản | Kênh | Nghĩa | Cú sốc | Nguồn con số |
|---|---|---|---:|---|
| `co_so` | chi_phi | giữ nguyên gánh nặng trả nợ hiện tại | ×1.0000 | mốc, không đổi |
| `chi_phi_bat_loi` | chi_phi | chi phí trả nợ bằng nhóm 25% nặng nhất của chính danh mục | ×1.2782 | p75 vs p50 đo trên dữ liệu |
| `chi_phi_rat_bat_loi` | chi_phi | chi phí trả nợ bằng nhóm 10% nặng nhất của chính danh mục | ×1.8996 | p90 vs p50 đo trên dữ liệu |
| `hanh_vi_bat_loi` | hanh_vi | hành vi trả nợ xấu đi bằng nhóm 25% nặng nhất của chính danh mục | +25 điểm phân vị | p75 vs p50 đo trên dữ liệu |
| `hanh_vi_rat_bat_loi` | hanh_vi | hành vi trả nợ xấu đi bằng nhóm 10% nặng nhất của chính danh mục | +40 điểm phân vị | p90 vs p50 đo trên dữ liệu |

## Tác động lên danh mục

Model cuối (ensemble stacking), **ngưỡng giữ nguyên 0.1173** ở mọi kịch bản — kịch bản làm dịch chuyển điểm số, không dịch chuyển chính sách. Danh mục: 246,008 hồ sơ (phần CV, không chạm holdout).

| Kịch bản | Kênh | Cú sốc | PD TB | PD trung vị | % bị chặn | Chênh so với cơ sở |
|---|---|---:|---:|---:|---:|---:|
| `co_so` | chi_phi | ×1.0000 | 0.0806 | 0.0483 | 21.7% | +0.0% |
| `chi_phi_bat_loi` | chi_phi | ×1.2782 | 0.0816 | 0.0506 | 22.2% | +0.5% |
| `chi_phi_rat_bat_loi` | chi_phi | ×1.8996 | 0.0780 | 0.0488 | 21.0% | -0.8% |
| `hanh_vi_bat_loi` | hanh_vi | +25đ phân vị | 0.1164 | 0.0612 | 30.9% | +9.2% |
| `hanh_vi_rat_bat_loi` | hanh_vi | +40đ phân vị | 0.1207 | 0.0660 | 32.4% | +10.7% |

## Phân bố nhóm rủi ro

| Kịch bản | PD < 5% | 5–15% | 15–30% | ≥ 30% |
|---|---:|---:|---:|---:|
| `co_so` | 51.2% | 33.2% | 12.0% | 3.6% |
| `chi_phi_bat_loi` | 49.6% | 34.6% | 12.4% | 3.4% |
| `chi_phi_rat_bat_loi` | 50.9% | 34.5% | 11.7% | 2.9% |
| `hanh_vi_bat_loi` | 43.7% | 31.8% | 14.6% | 9.9% |
| `hanh_vi_rat_bat_loi` | 41.1% | 33.3% | 15.3% | 10.3% |

Biểu đồ: `outputs/plots/ph1_stress_bands.png`.

## Giới hạn

- **Phân tích độ nhạy, không phải dự báo.** "nếu gánh nặng trả nợ tăng lên mức X thì model chấm lại ra sao", không trả lời "khi lãi suất tăng 2% thì bao nhiêu người vỡ nợ".
- **Chỉ 3 cột đổi:** `AMT_ANNUITY`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`. Mọi cột lịch sử (bureau/previous/installments) giữ nguyên
- **Không tách được lãi suất khỏi kỳ hạn:** `CREDIT_TERM` cao có thể do vay đắt hoặc phải trả nhanh. Nên đây là cú sốc **gánh nặng trả nợ**, không phải cú sốc lãi suất thuần tuý.
- **Hành vi khách không nằm trong model:** vay thêm chỗ khác, cơ cấu nợ, bỏ nợ đều không được mô hình hoá.
- **Kênh hành vi có tính vòng quanh nhẹ:** trễ hạn ở khoản vay *trước* là dự báo mạnh cho vỡ nợ ở khoản vay *hiện tại*, nên giả định "khách bắt đầu trả trễ" rồi kết luận "họ dễ vỡ nợ" là một lập luận gần với đồng nghĩa. Nó trả lời "nếu chất lượng trả nợ của danh mục tụt xuống mức của nhóm xấu hơn thì điểm số dịch chuyển ra sao", không trả lời "cú sốc vĩ mô làm bao nhiêu người trả trễ".
- **Kênh hành vi sốc chưa trọn vẹn:** các cột trễ hạn bị đẩy xấu đi, nhưng cột trả đúng hạn/trả đủ (`INST_PAYMENT_PERC_MEAN`, `INST_DBD_*`) giữ nguyên, nên hồ sơ sau cú sốc hơi thiếu nhất quán nội tại.
- **`CC_UTILIZATION_*` chỉ phủ ~25% danh mục** (75% khách không có thẻ tín dụng), nên phần đóng góp của cột này vào cú sốc nhỏ hơn vẻ ngoài.
