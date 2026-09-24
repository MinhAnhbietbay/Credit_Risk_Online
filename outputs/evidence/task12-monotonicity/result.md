stress-test chạy trên chi phí trả nợ (`AMT_ANNUITY`) và hành vi trả nợ (trễ hạn / dùng cạn hạn mức) — vì kênh chi phí một mình cho kết quả sai chiều.

**So sánh:** tỉ lệ vỡ nợ theo thập phân vị của từng cột trên phần CV. Cột đơn điệu tăng thì đẩy nó lên cao mới thực sự là 'xấu đi' dưới mắt model.

**Số:** KHÔNG đơn điệu: CREDIT_TERM, AMT_ANNUITY, ANNUITY_INCOME_RATIO. Đơn điệu: INST_LATE_RATIO, INST_DPD_MEAN, INST_DPD_STD, POS_SK_DPD_DEF_MEAN, CC_UTILIZATION_MEAN, CC_UTILIZATION_MAX. Chi tiết từng thập phân vị ở `monotonicity.csv`.

`CREDIT_TERM` và `AMT_ANNUITY` — hai cột của kênh chi phí — có thập phân vị **cao nhất** lại là nhóm **ít vỡ nợ nhất**, vì đó là vay tiêu dùng POS món nhỏ kỳ hạn rất ngắn, một phân khúc khách khác hẳn. Đẩy cả danh mục vào vùng đó không mô phỏng cú sốc — nó mô phỏng 'giả vờ mọi khách đều là khách vay ngắn hạn'.
