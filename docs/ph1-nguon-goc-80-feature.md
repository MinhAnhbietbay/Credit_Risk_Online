# Nguồn gốc 80 feature của model

Tài liệu này trả lời: với mỗi feature trong `data/processed/selected_features.json` (80 feature), nó được tính **từ cột gốc nào, ở bảng nào, bằng cách nào**. Mọi nội dung lấy từ code trong `src/ph1_credit_risk/features/` và đối chiếu với tên cột trong các file CSV gốc của Home Credit. Không có số nào ở đây do ước đoán.

## 1. Tổng quan

Dataset gốc có 7 bảng. Bảng `application_train.csv` có **mỗi khách một dòng**. Các bảng còn lại có **mỗi khách nhiều dòng** (mỗi dòng là một khoản vay cũ, một kỳ trả góp hay một tháng dư nợ), nên pipeline phải gộp (mean, max, sum…) về một dòng cho mỗi khách rồi ghép vào bảng chính theo `SK_ID_CURR`.

| Bảng nguồn | Số feature trong 80 | Trong đó cột gốc dùng trực tiếp | Trong đó tự tính | Số cột gốc khác nhau cần có |
|---|---|---|---|---|
| `application_train` (đơn vay hiện tại) | 23 | 16 | 7 | 17 |
| `bureau` (khoản vay ở tổ chức khác) | 19 | 0 | 19 | 7 |
| `previous_application` (đơn vay cũ ở Home Credit) | 18 | 0 | 18 | 11 |
| `installments_payments` (lịch sử trả góp) | 14 | 0 | 14 | 5 |
| `credit_card_balance` (thẻ tín dụng) | 4 | 0 | 4 | 4 |
| `POS_CASH_balance` (vay trả góp POS) | 2 | 0 | 2 | 2 |
| **Tổng** | **80** | **16** | **64** | **46** |

Ghi chú:
- "Cột gốc khác nhau cần có" là số cột thô (không tính khóa `SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`) mà bạn phải có để tính ra toàn bộ feature của bảng đó. Nhiều feature có thể dùng chung một cột gốc (ví dụ `DAYS_CREDIT` sinh ra nhiều feature).
- Ở 5 bảng phụ, **không feature nào là cột gốc dùng thẳng**; tất cả đều là kết quả gộp nhiều dòng.
- Cột gốc của bảng đơn vay là 17 vì ngoài 16 cột được dùng trực tiếp còn cần `AMT_INCOME_TOTAL` để tính hai tỉ lệ thu nhập (cột này bản thân không nằm trong 80 feature).

### Quy ước đọc các bảng bên dưới

- **Gộp** nghĩa là tính trên tất cả dòng của cùng một khách: `MEAN` trung bình, `SUM` tổng, `MAX` lớn nhất, `MIN` nhỏ nhất, `STD` độ lệch chuẩn.
- Cột `DAYS_*` đo số ngày **so với ngày nộp đơn hiện tại**, nên giá trị âm nghĩa là quá khứ (ví dụ `-365` là 1 năm trước).
- Cột `MONTHS_BALANCE` tương tự nhưng tính bằng tháng.
- Khách nào không có dòng nào trong một bảng phụ thì mọi feature của bảng đó là **trống (NaN)**, vì pipeline ghép bằng `left join` (`join_features` trong `builder.py`).
- Một số cột chữ (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, `NAME_EDUCATION_TYPE`, `PREV_PRODUCT_COMBINATION_MODE`) được mã hóa thành số nguyên trong `builder.py` (`encode_categoricals`).

---

## 2. Bảng `application_train` (23 feature)

Mỗi khách đúng một dòng, nên không cần gộp.

### 2.1. 16 feature là cột gốc

| Feature | Cột gốc | Ý nghĩa |
|---|---|---|
| `AMT_ANNUITY` | `AMT_ANNUITY` | Số tiền phải trả mỗi kỳ |
| `AMT_CREDIT` | `AMT_CREDIT` | Số tiền vay |
| `DAYS_BIRTH` | `DAYS_BIRTH` | Số ngày tính từ ngày sinh đến lúc nộp đơn (âm) |
| `DAYS_EMPLOYED` | `DAYS_EMPLOYED` | Số ngày đã đi làm ở công việc hiện tại (âm). Giá trị đặc biệt `365243` đổi thành trống (`add_application_features`) |
| `DAYS_ID_PUBLISH` | `DAYS_ID_PUBLISH` | Số ngày kể từ lần đổi giấy tờ tùy thân gần nhất |
| `DAYS_REGISTRATION` | `DAYS_REGISTRATION` | Số ngày kể từ lần đổi đăng ký cư trú gần nhất |
| `DAYS_LAST_PHONE_CHANGE` | `DAYS_LAST_PHONE_CHANGE` | Số ngày kể từ lần đổi số điện thoại gần nhất |
| `EXT_SOURCE_1` | `EXT_SOURCE_1` | Điểm tín dụng nguồn ngoài số 1 (0 đến 1, càng cao càng tốt) |
| `EXT_SOURCE_2` | `EXT_SOURCE_2` | Điểm tín dụng nguồn ngoài số 2 |
| `EXT_SOURCE_3` | `EXT_SOURCE_3` | Điểm tín dụng nguồn ngoài số 3 |
| `NAME_EDUCATION_TYPE` | `NAME_EDUCATION_TYPE` | Trình độ học vấn cao nhất (danh mục) |
| `OCCUPATION_TYPE` | `OCCUPATION_TYPE` | Nghề nghiệp (danh mục) |
| `ORGANIZATION_TYPE` | `ORGANIZATION_TYPE` | Loại tổ chức nơi làm việc (danh mục) |
| `OWN_CAR_AGE` | `OWN_CAR_AGE` | Tuổi xe ô tô khách sở hữu (năm) |
| `REGION_POPULATION_RELATIVE` | `REGION_POPULATION_RELATIVE` | Mật độ dân cư chuẩn hóa của nơi khách ở |
| `YEARS_BEGINEXPLUATATION_AVG` | `YEARS_BEGINEXPLUATATION_AVG` | Chỉ số chuẩn hóa về thời gian đưa tòa nhà vào sử dụng (tên cột gốc viết sai chính tả như vậy) |

### 2.2. 7 feature tự tính từ chính dòng này

Nguồn: `add_application_features` trong `features/application.py`. Phép chia nào có mẫu bằng 0 thì cho kết quả trống.

| Feature | Cột gốc cần | Cách tính |
|---|---|---|
| `CREDIT_TERM` | `AMT_ANNUITY`, `AMT_CREDIT` | `AMT_ANNUITY / AMT_CREDIT` (nghịch đảo của số kỳ trả ước tính) |
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT`, `AMT_INCOME_TOTAL` | `AMT_CREDIT / AMT_INCOME_TOTAL` |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY`, `AMT_INCOME_TOTAL` | `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| `EXT_SOURCE_MEAN` | `EXT_SOURCE_1/2/3` | Trung bình của 3 điểm (bỏ qua điểm trống) |
| `EXT_SOURCE_MIN` | `EXT_SOURCE_1/2/3` | Nhỏ nhất của 3 điểm |
| `EXT_SOURCE_MAX` | `EXT_SOURCE_1/2/3` | Lớn nhất của 3 điểm |
| `EXT_SOURCE_STD` | `EXT_SOURCE_1/2/3` | Độ lệch chuẩn của 3 điểm |

---

## 3. Bảng `bureau`: khoản vay ở tổ chức tín dụng khác (19 feature)

Mỗi khách có nhiều khoản vay; mỗi khoản là một dòng. Code: `features/bureau.py`, hàm `agg_bureau`. Có hai nhóm: gộp trên **mọi** khoản (`BUREAU_*`) và gộp chỉ trên khoản **đang hoạt động** (`BUREAU_ACTIVE_*`, tức `CREDIT_ACTIVE = "Active"`).

Cột gốc trong bảng `bureau`:

| Cột gốc | Ý nghĩa |
|---|---|
| `DAYS_CREDIT` | Khách nộp đơn vay khoản này cách ngày nộp đơn hiện tại bao nhiêu ngày (âm) |
| `DAYS_CREDIT_ENDDATE` | Số ngày còn lại đến hạn kết thúc khoản vay, tính tại lúc nộp đơn hiện tại |
| `DAYS_ENDDATE_FACT` | Số ngày kể từ khi khoản vay thực tế kết thúc (chỉ có với khoản đã đóng) |
| `AMT_CREDIT_SUM` | Tổng số tiền/hạn mức của khoản vay |
| `AMT_CREDIT_SUM_DEBT` | Dư nợ hiện tại của khoản vay |
| `AMT_CREDIT_MAX_OVERDUE` | Số tiền quá hạn tối đa từng có trên khoản vay |
| `CREDIT_ACTIVE` | Trạng thái khoản vay (`Active`, `Closed`, `Sold`, `Bad debt`) |

### 3.1. Gộp trên mọi khoản vay (10 feature)

| Feature | Cột gốc | Cách tính (trên các khoản vay của khách) |
|---|---|---|
| `BUREAU_DAYS_CREDIT_MAX` | `DAYS_CREDIT` | Giá trị lớn nhất, tức khoản vay **gần đây nhất** (âm nhỏ nhất) |
| `BUREAU_DAYS_CREDIT_MEAN` | `DAYS_CREDIT` | Trung bình |
| `BUREAU_DAYS_CREDIT_ENDDATE_MAX` | `DAYS_CREDIT_ENDDATE` | Lớn nhất |
| `BUREAU_DAYS_ENDDATE_FACT_MAX` | `DAYS_ENDDATE_FACT` | Lớn nhất |
| `BUREAU_AMT_CREDIT_SUM_SUM` | `AMT_CREDIT_SUM` | Tổng |
| `BUREAU_AMT_CREDIT_SUM_MEAN` | `AMT_CREDIT_SUM` | Trung bình |
| `BUREAU_AMT_CREDIT_SUM_MAX` | `AMT_CREDIT_SUM` | Lớn nhất |
| `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN` | `AMT_CREDIT_SUM_DEBT` | Trung bình |
| `BUREAU_AMT_CREDIT_MAX_OVERDUE_SUM` | `AMT_CREDIT_MAX_OVERDUE` | Tổng |
| `BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT` | `CREDIT_ACTIVE` | **Đếm** số khoản vay có trạng thái `Active` |

### 3.2. Gộp chỉ trên khoản đang hoạt động (9 feature)

Đầu tiên lọc các khoản có `CREDIT_ACTIVE = "Active"`, rồi mới gộp.

| Feature | Cột gốc | Cách tính |
|---|---|---|
| `BUREAU_ACTIVE_DAYS_CREDIT_MAX` | `DAYS_CREDIT` | Lớn nhất |
| `BUREAU_ACTIVE_DAYS_CREDIT_MEAN` | `DAYS_CREDIT` | Trung bình |
| `BUREAU_ACTIVE_DAYS_CREDIT_ENDDATE_MIN` | `DAYS_CREDIT_ENDDATE` | Nhỏ nhất |
| `BUREAU_ACTIVE_DAYS_CREDIT_ENDDATE_MEAN` | `DAYS_CREDIT_ENDDATE` | Trung bình |
| `BUREAU_ACTIVE_DAYS_CREDIT_ENDDATE_MAX` | `DAYS_CREDIT_ENDDATE` | Lớn nhất |
| `BUREAU_ACTIVE_AMT_CREDIT_SUM_MIN` | `AMT_CREDIT_SUM` | Nhỏ nhất |
| `BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_MIN` | `AMT_CREDIT_SUM_DEBT` | Nhỏ nhất |
| `BUREAU_ACTIVE_AMT_CREDIT_MAX_OVERDUE_SUM` | `AMT_CREDIT_MAX_OVERDUE` | Tổng |
| `BUREAU_ACTIVE_RATIO` | `CREDIT_ACTIVE` | `BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT / BUREAU_COUNT`, tức số khoản đang hoạt động chia tổng số khoản vay (`BUREAU_COUNT` là số dòng của khách) |

Lưu ý về tên: `BUREAU_ACTIVE_RATIO` **không** thuộc nhóm "chỉ tính khoản đang hoạt động" dù tên có `ACTIVE`; nó tính trên mọi khoản.

---

## 4. Bảng `previous_application`: đơn vay cũ ở Home Credit (18 feature)

Mỗi khách có nhiều đơn vay cũ; mỗi đơn là một dòng. Code: `features/previous.py`, hàm `agg_previous`. Có ba nhóm: gộp trên mọi đơn (`PREV_*`), trên đơn đã **được duyệt** (`PREV_APPROVED_*`), trên đơn **bị từ chối** (`PREV_REFUSED_*`, riêng `PREV_REFUSED_RATIO` không thuộc nhóm lọc).

Cột gốc trong bảng `previous_application`:

| Cột gốc | Ý nghĩa |
|---|---|
| `NAME_CONTRACT_STATUS` | Kết quả đơn (`Approved`, `Refused`, `Canceled`, `Unused offer`) |
| `AMT_APPLICATION` | Số tiền khách xin vay |
| `AMT_CREDIT` | Số tiền cuối cùng được cấp |
| `AMT_ANNUITY` | Số tiền trả mỗi kỳ của đơn đó |
| `AMT_DOWN_PAYMENT` | Số tiền trả trước |
| `CNT_PAYMENT` | Kỳ hạn của đơn (số kỳ trả) |
| `DAYS_DECISION` | Quyết định duyệt đơn cách ngày nộp đơn hiện tại bao nhiêu ngày (âm) |
| `DAYS_LAST_DUE` | Ngày đến hạn trả cuối cùng thực tế. `365243` đổi thành trống |
| `DAYS_LAST_DUE_1ST_VERSION` | Ngày đến hạn trả cuối cùng theo phiên bản hợp đồng đầu tiên. `365243` đổi thành trống |
| `HOUR_APPR_PROCESS_START` | Giờ trong ngày khách nộp đơn (làm tròn) |
| `PRODUCT_COMBINATION` | Tổ hợp sản phẩm của đơn (danh mục) |

Có thêm một cột **tự tính ở mức từng đơn** trước khi gộp: `APP_CREDIT_RATIO = AMT_APPLICATION / AMT_CREDIT` (xin bao nhiêu trên được cấp bao nhiêu; `AMT_CREDIT = 0` cho kết quả trống).

### 4.1. Gộp trên mọi đơn (11 feature)

| Feature | Cột gốc | Cách tính |
|---|---|---|
| `PREV_DAYS_DECISION_MIN` | `DAYS_DECISION` | Nhỏ nhất, tức đơn **xa nhất** trong quá khứ |
| `PREV_DAYS_LAST_DUE_MAX` | `DAYS_LAST_DUE` | Lớn nhất |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | `DAYS_LAST_DUE_1ST_VERSION` | Lớn nhất |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN` | `DAYS_LAST_DUE_1ST_VERSION` | Trung bình |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_SUM` | `DAYS_LAST_DUE_1ST_VERSION` | Tổng |
| `PREV_CNT_PAYMENT_MEAN` | `CNT_PAYMENT` | Trung bình kỳ hạn |
| `PREV_HOUR_APPR_PROCESS_START_MEAN` | `HOUR_APPR_PROCESS_START` | Trung bình giờ nộp đơn |
| `PREV_AMT_DOWN_PAYMENT_SUM` | `AMT_DOWN_PAYMENT` | Tổng tiền trả trước |
| `PREV_APP_CREDIT_RATIO_MEAN` | `AMT_APPLICATION`, `AMT_CREDIT` | Trung bình của `AMT_APPLICATION / AMT_CREDIT` theo từng đơn |
| `PREV_APP_CREDIT_RATIO_MIN` | `AMT_APPLICATION`, `AMT_CREDIT` | Nhỏ nhất của tỉ lệ trên |
| `PREV_PRODUCT_COMBINATION_MODE` | `PRODUCT_COMBINATION` | Giá trị xuất hiện **nhiều nhất** (mode); hòa thì lấy giá trị đứng trước theo thứ tự sắp xếp |

### 4.2. Gộp chỉ trên đơn đã được duyệt, `NAME_CONTRACT_STATUS = "Approved"` (4 feature)

| Feature | Cột gốc | Cách tính |
|---|---|---|
| `PREV_APPROVED_AMT_ANNUITY_SUM` | `AMT_ANNUITY` | Tổng |
| `PREV_APPROVED_AMT_ANNUITY_MEAN` | `AMT_ANNUITY` | Trung bình |
| `PREV_APPROVED_AMT_APPLICATION_SUM` | `AMT_APPLICATION` | Tổng |
| `PREV_APPROVED_AMT_DOWN_PAYMENT_SUM` | `AMT_DOWN_PAYMENT` | Tổng |

### 4.3. Liên quan đơn bị từ chối (3 feature)

| Feature | Cột gốc | Cách tính |
|---|---|---|
| `PREV_REFUSED_RATIO` | `NAME_CONTRACT_STATUS` | Số đơn `Refused` chia tổng số đơn cũ của khách (`PREV_COUNT`). Tính trên **mọi** đơn |
| `PREV_REFUSED_DAYS_DECISION_MAX` | `DAYS_DECISION` | Chỉ trên đơn `Refused`: lớn nhất, tức lần bị từ chối **gần đây nhất** |
| `PREV_REFUSED_APP_CREDIT_RATIO_MIN` | `AMT_APPLICATION`, `AMT_CREDIT` | Chỉ trên đơn `Refused`: nhỏ nhất của `AMT_APPLICATION / AMT_CREDIT` |

---

## 5. Bảng `installments_payments`: lịch sử trả góp (14 feature)

Mỗi dòng là một kỳ trả góp đã đến hạn của một khoản vay cũ. Code: `features/installments.py`. Bảng này không có `SK_ID_CURR` trực tiếp; nó được nối với khách qua `SK_ID_PREV` bằng bảng `prev_id_map` (lấy từ `previous_application`).

Cột gốc trong bảng `installments_payments`:

| Cột gốc | Ý nghĩa |
|---|---|
| `DAYS_INSTALMENT` | Ngày đến hạn theo lịch của kỳ trả |
| `DAYS_ENTRY_PAYMENT` | Ngày khách thực sự trả |
| `AMT_INSTALMENT` | Số tiền phải trả theo lịch |
| `AMT_PAYMENT` | Số tiền thực trả |
| `NUM_INSTALMENT_NUMBER` | Số thứ tự kỳ trả trong khoản vay |

Bốn cột **tự tính cho từng kỳ trả** trước khi gộp (hàm `add_row_features`):

| Cột tự tính theo kỳ | Công thức |
|---|---|
| `DPD` (số ngày trễ) | `max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)` |
| `DBD` (số ngày trả sớm) | `max(0, DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT)` |
| `PAYMENT_PERC` | `AMT_PAYMENT / AMT_INSTALMENT` (mẫu bằng 0 cho kết quả trống) |
| `PAYMENT_DIFF` | `AMT_INSTALMENT - AMT_PAYMENT` (dương nghĩa là trả thiếu) |

| Feature | Cột gốc | Cách tính (trên các kỳ trả của khách) |
|---|---|---|
| `INST_LATE_RATIO` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | **Số kỳ có `DPD > 0` chia tổng số kỳ**, tức tỉ lệ kỳ trả bị trễ |
| `INST_DPD_MEAN` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | Trung bình `DPD` |
| `INST_DPD_STD` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | Độ lệch chuẩn `DPD` |
| `INST_DBD_MEAN` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | Trung bình `DBD` |
| `INST_DBD_MAX` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | Lớn nhất `DBD` |
| `INST_DBD_SUM` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | Tổng `DBD` |
| `INST_DBD_STD` | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | Độ lệch chuẩn `DBD` |
| `INST_PAYMENT_PERC_MEAN` | `AMT_PAYMENT`, `AMT_INSTALMENT` | Trung bình `PAYMENT_PERC` |
| `INST_PAYMENT_DIFF_MEAN` | `AMT_INSTALMENT`, `AMT_PAYMENT` | Trung bình `PAYMENT_DIFF` |
| `INST_AMT_INSTALMENT_SUM` | `AMT_INSTALMENT` | Tổng |
| `INST_AMT_INSTALMENT_MEAN` | `AMT_INSTALMENT` | Trung bình |
| `INST_AMT_INSTALMENT_MAX` | `AMT_INSTALMENT` | Lớn nhất |
| `INST_DAYS_ENTRY_PAYMENT_MAX` | `DAYS_ENTRY_PAYMENT` | Lớn nhất, tức lần trả **gần đây nhất** |
| `INST_NUM_INSTALMENT_NUMBER_MAX` | `NUM_INSTALMENT_NUMBER` | Lớn nhất, tức số kỳ trả lớn nhất ở một khoản vay nào đó |

Ví dụ `INST_LATE_RATIO`: khách có 40 kỳ trả trong lịch sử, 6 kỳ trả trễ hơn ngày đến hạn thì `INST_LATE_RATIO = 6 / 40 = 0.15`.

---

## 6. Bảng `credit_card_balance`: thẻ tín dụng (4 feature)

Mỗi dòng là một tháng dư nợ của một thẻ. Code: `features/credit_card.py`. Nối với khách qua `SK_ID_PREV` như bảng trả góp.

Cột gốc trong bảng `credit_card_balance`:

| Cột gốc | Ý nghĩa |
|---|---|
| `AMT_BALANCE` | Dư nợ thẻ cuối tháng |
| `AMT_CREDIT_LIMIT_ACTUAL` | Hạn mức thẻ trong tháng |
| `CNT_DRAWINGS_ATM_CURRENT` | Số lần rút tiền mặt qua ATM trong tháng |
| `CNT_DRAWINGS_CURRENT` | Tổng số lần rút/giao dịch trong tháng |

Cột tự tính theo tháng: `UTILIZATION = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` (hạn mức 0 cho kết quả trống).

| Feature | Cột gốc | Cách tính (trên các tháng của khách) |
|---|---|---|
| `CC_UTILIZATION_MEAN` | `AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL` | Trung bình tỉ lệ dùng hạn mức |
| `CC_UTILIZATION_MAX` | `AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL` | Lớn nhất tỉ lệ dùng hạn mức |
| `CC_CNT_DRAWINGS_ATM_CURRENT_MEAN` | `CNT_DRAWINGS_ATM_CURRENT` | Trung bình số lần rút ATM mỗi tháng |
| `CC_CNT_DRAWINGS_CURRENT_MEAN` | `CNT_DRAWINGS_CURRENT` | Trung bình số lần giao dịch mỗi tháng |

---

## 7. Bảng `POS_CASH_balance`: vay trả góp POS (2 feature)

Mỗi dòng là một tháng của một khoản vay POS/tiền mặt. Code: `features/pos_cash.py`. Nối với khách qua `SK_ID_PREV`.

| Feature | Cột gốc | Cách tính (trên các tháng của khách) |
|---|---|---|
| `POS_SK_DPD_DEF_MEAN` | `SK_DPD_DEF` | Trung bình số ngày quá hạn (có ngưỡng bỏ qua trễ nhỏ) mỗi tháng |
| `POS_MONTHS_BALANCE_SIZE` | `MONTHS_BALANCE` | **Đếm số dòng** của khách, tức tổng số tháng có ghi nhận |

---

## 8. Điều cần lưu ý khi dùng vào web

1. **Tổng số cột gốc cần có là 46**, nhưng chỉ 17 cột nằm ở bảng đơn vay (một dòng, nhập được một lần). 29 cột còn lại nằm ở 5 bảng phụ và mỗi cột là **nhiều giá trị** của một khách (nhiều khoản vay, nhiều kỳ trả, nhiều tháng thẻ). Muốn tính đúng các feature này, hệ thống cần dữ liệu từng dòng lịch sử (từ hệ thống ngân hàng hoặc CIC), không phải một ô nhập.
2. **Khách chưa có lịch sử ở một bảng phụ** có feature trống (NaN) khi train, vì ghép bằng `left join`. Form web hiện điền **trung vị** cho feature không hỏi (xem `build_feature_row` trong `src/api/services/scoring.py`). Hai cách xử lý này khác nhau: trung vị giả định khách có lịch sử "trung bình", còn NaN nghĩa là "không có lịch sử". Tôi **chưa kiểm** cách model xử lý NaN (LightGBM/CatBoost xử lý được, còn mô hình tuyến tính trong ensemble thì cần xem pipeline có điền giá trị trước không), nên chưa biết sự khác nhau này ảnh hưởng kết quả bao nhiêu.
3. **Bảy feature ở mục 2.2 tính được ngay** từ cột gốc của đơn vay hiện tại, nên form chỉ cần hỏi cột gốc (`EXT_SOURCE_1/2/3`, `AMT_ANNUITY`, `AMT_CREDIT`, `AMT_INCOME_TOTAL`) rồi để backend tính, thay vì bắt người dùng tự nhập `EXT_SOURCE_MEAN`, `CREDIT_TERM`… như hiện nay.
4. **Mười bốn feature `INST_*` có thể tính từ danh sách kỳ trả** nếu form cho nhập lịch sử từng kỳ (ngày đến hạn, ngày thực trả, số tiền). Đây là hướng khả thi nếu cần giữ độ chính xác cho nhóm feature quan trọng này (`INST_LATE_RATIO` đứng thứ 5 theo SHAP).
