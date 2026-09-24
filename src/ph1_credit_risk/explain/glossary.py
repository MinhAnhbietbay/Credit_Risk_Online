"""Từ điển mô tả feature.
Tên không tra được thì `describe()` trả lại chính tên thô và `has_description()` trả False — không bịa nghĩa.
"""

from __future__ import annotations
import json

__all__ = ["describe", "has_description", "decode_category", "TABLE", "FILTER", "AGG", "BASE", "WHOLE_NAME"]

# Tiền tố bảng nguồn
TABLE = {
    "BUREAU": "CIC",                       # bureau.csv — khoản vay ở tổ chức tín dụng khác
    "PREV": "đơn vay trước ở Home Credit",  # previous_application.csv
    "INST": "lịch sử trả góp",             # installments_payments.csv
    "POS": "khoản trả góp POS/tiền mặt",   # POS_CASH_balance.csv
    "CC": "thẻ tín dụng",                  # credit_card_balance.csv
}

# Bộ lọc tập con (đứng ngay sau tiền tố bảng) 
FILTER = {
    "ACTIVE": "tính trên khoản còn dư nợ",
    "APPROVED": "tính trên đơn đã duyệt",
    "REFUSED": "tính trên đơn bị từ chối",
}

# Phép tổng hợp (hậu tố)
AGG = {
    "MEAN": "trung bình",
    "MAX": "lớn nhất",
    "MIN": "nhỏ nhất",
    "SUM": "tổng",
    "STD": "độ lệch chuẩn",
    "COUNT": "số lượng",
    "SIZE": "số bản ghi",
    "MODE": "giá trị hay gặp nhất",
    "LAST": "lần gần nhất",
}

# Cột gốc
BASE = {
    # application_train — nhân thân, thu nhập, khoản vay đang xin
    "AMT_ANNUITY": "số tiền phải trả mỗi kỳ",
    "AMT_CREDIT": "số tiền vay được duyệt",
    "AMT_APPLICATION": "số tiền khách xin vay",
    "AMT_INCOME_TOTAL": "thu nhập khai báo",
    "AMT_DOWN_PAYMENT": "số tiền trả trước",
    "AMT_GOODS_PRICE": "giá món hàng được tài trợ",
    "CODE_GENDER": "giới tính",
    "DAYS_BIRTH": "số ngày tính đến ngày sinh (âm; càng âm càng lớn tuổi)",
    "DAYS_EMPLOYED": "số ngày kể từ khi vào làm công việc hiện tại (âm = quá khứ)",
    "DAYS_ID_PUBLISH": "số ngày kể từ lần đổi giấy tờ tùy thân gần nhất (âm = quá khứ)",
    "DAYS_REGISTRATION": "số ngày kể từ lần thay đổi đăng ký cư trú gần nhất (âm = quá khứ)",
    "DAYS_LAST_PHONE_CHANGE": "số ngày kể từ lần đổi số điện thoại gần nhất (âm = quá khứ)",
    "NAME_EDUCATION_TYPE": "trình độ học vấn cao nhất",
    "NAME_FAMILY_STATUS": "tình trạng hôn nhân",
    "NAME_CONTRACT_TYPE": "loại hợp đồng vay (tiền mặt hay quay vòng)",
    "OCCUPATION_TYPE": "nghề nghiệp",
    "ORGANIZATION_TYPE": "loại tổ chức nơi khách làm việc",
    "OWN_CAR_AGE": "tuổi xe ô tô khách sở hữu (năm)",
    "REGION_POPULATION_RELATIVE": "mật độ dân cư khu vực khách sinh sống (đã chuẩn hóa)",
    # Nhóm cột "toà nhà nơi khách ở" của application_train: hậu tố _AVG/_MODE/_MEDI là trung bình/mode/trung vị
    # các chỉ số toà nhà, đã chuẩn hóa về 0-1 (mô tả gốc của Kaggle gộp chung cả nhóm).
    "YEARS_BEGINEXPLUATATION_AVG": "tuổi toà nhà nơi khách ở, chuẩn hóa 0-1 (cao = toà nhà mới đưa vào sử dụng gần đây)",
    "EXT_SOURCE_1": "điểm tín dụng từ nguồn ngoài số 1 (0-1, càng cao càng tốt)",
    "EXT_SOURCE_2": "điểm tín dụng từ nguồn ngoài số 2 (0-1, càng cao càng tốt)",
    "EXT_SOURCE_3": "điểm tín dụng từ nguồn ngoài số 3 (0-1, càng cao càng tốt)",
    "CNT_CHILDREN": "số con",
    "CNT_FAM_MEMBERS": "số thành viên gia đình",
    "HOUR_APPR_PROCESS_START": "giờ trong ngày lúc nộp đơn",

    # bureau.csv
    "DAYS_CREDIT": "số ngày kể từ khi mở khoản vay ở tổ chức khác (âm = quá khứ)",
    "DAYS_CREDIT_ENDDATE": "số ngày đến hạn tất toán theo hợp đồng (dương = còn hạn ở tương lai)",
    "DAYS_ENDDATE_FACT": "số ngày kể từ khi tất toán thực tế (âm = đã tất toán)",
    "DAYS_CREDIT_UPDATE": "số ngày kể từ lần CIC cập nhật gần nhất (âm = quá khứ)",
    "AMT_CREDIT_SUM": "hạn mức/tổng dư nợ gốc của khoản vay",
    "AMT_CREDIT_SUM_DEBT": "dư nợ còn lại",
    "AMT_CREDIT_SUM_OVERDUE": "số tiền đang quá hạn",
    "AMT_CREDIT_MAX_OVERDUE": "số tiền quá hạn cao nhất từng ghi nhận",
    "CNT_CREDIT_PROLONG": "số lần được gia hạn khoản vay",
    "CREDIT_DAY_OVERDUE": "số ngày đang quá hạn",
    "CREDIT_ACTIVE_ACTIVE": "khoản vay đang ở trạng thái còn dư nợ",
    "CREDIT_ACTIVE_CLOSED": "khoản vay đã tất toán",
    "MONTHS_BALANCE": "số tháng tính ngược từ thời điểm nộp đơn (âm = quá khứ)",

    # previous_application.csv
    "DAYS_DECISION": "số ngày kể từ khi có quyết định cho đơn vay trước (âm = quá khứ)",
    "DAYS_LAST_DUE": "số ngày đến kỳ trả cuối theo hợp đồng (âm = đã qua)",
    "DAYS_LAST_DUE_1ST_VERSION": "số ngày đến kỳ trả cuối theo lịch ký ban đầu (lệch nhiều so với thực tế = trả sớm/trễ)",
    "DAYS_FIRST_DUE": "số ngày đến kỳ trả đầu tiên (âm = đã qua)",
    "DAYS_TERMINATION": "số ngày đến khi hợp đồng kết thúc (âm = đã kết thúc)",
    "CNT_PAYMENT": "số kỳ trả của khoản vay trước",
    "RATE_DOWN_PAYMENT": "tỉ lệ trả trước",
    "PRODUCT_COMBINATION": "gói sản phẩm vay",
    "STATUS_REFUSED": "đơn bị từ chối",
    "STATUS_APPROVED": "đơn được duyệt",
    "NAME_CONTRACT_STATUS": "trạng thái xét duyệt đơn",

    # installments_payments.csv (DPD/DBD/PAYMENT_* do features/installments.py sinh)
    "AMT_INSTALMENT": "số tiền phải trả theo lịch của một kỳ",
    "AMT_PAYMENT": "số tiền khách thực trả trong kỳ",
    "NUM_INSTALMENT_NUMBER": "số thứ tự kỳ trả",
    "NUM_INSTALMENT_VERSION": "phiên bản lịch trả (đổi = lịch bị sửa)",
    "DAYS_ENTRY_PAYMENT": "số ngày kể từ lúc khách thực trả (âm = quá khứ)",
    "DAYS_INSTALMENT": "số ngày kể từ hạn trả theo lịch (âm = quá khứ)",
    "DPD": "số ngày trả trễ so với hạn (0 nếu không trễ)",
    "DBD": "số ngày trả sớm trước hạn (0 nếu không sớm)",
    "PAYMENT_PERC": "tỉ lệ thực trả trên số phải trả của kỳ (1.0 = trả đủ)",
    "PAYMENT_DIFF": "phần thiếu của kỳ = số phải trả trừ số thực trả (dương = trả thiếu)",

    # POS_CASH_balance.csv
    "SK_DPD": "số ngày quá hạn trong tháng",
    "SK_DPD_DEF": "số ngày quá hạn trong tháng, bỏ qua các khoản trễ không đáng kể",
    "CNT_INSTALMENT": "tổng số kỳ trả của hợp đồng",
    "CNT_INSTALMENT_FUTURE": "số kỳ trả còn lại",

    # credit_card_balance.csv (UTILIZATION do features/credit_card.py sinh)
    "AMT_BALANCE": "dư nợ thẻ trong tháng",
    "AMT_CREDIT_LIMIT_ACTUAL": "hạn mức thẻ",
    "AMT_DRAWINGS_ATM_CURRENT": "số tiền rút ATM trong tháng",
    "AMT_DRAWINGS_CURRENT": "tổng số tiền rút/chi tiêu trong tháng",
    "AMT_PAYMENT_CURRENT": "số tiền trả thẻ trong tháng",
    "CNT_DRAWINGS_ATM_CURRENT": "số lần rút tiền mặt ở ATM trong tháng",
    "CNT_DRAWINGS_CURRENT": "tổng số lần rút/chi tiêu trong tháng",
    "UTILIZATION": "tỉ lệ sử dụng hạn mức thẻ = dư nợ / hạn mức",
}

# Feature dẫn xuất không theo cấu trúc tiền tố/hậu tố
WHOLE_NAME = {
    "CREDIT_INCOME_RATIO": "số tiền vay gấp bao nhiêu lần thu nhập năm",
    "ANNUITY_INCOME_RATIO": "tiền trả mỗi kỳ chiếm bao nhiêu phần thu nhập",
    "CREDIT_TERM": "tiền trả mỗi kỳ chia cho tổng khoản vay (nghịch đảo số kỳ ước tính)",
    "EXT_SOURCE_MEAN": "trung bình 3 điểm tín dụng nguồn ngoài (càng cao càng tốt)",
    "EXT_SOURCE_MIN": "điểm tín dụng nguồn ngoài thấp nhất trong 3 nguồn",
    "EXT_SOURCE_MAX": "điểm tín dụng nguồn ngoài cao nhất trong 3 nguồn",
    "EXT_SOURCE_STD": "mức chênh lệch giữa 3 điểm tín dụng nguồn ngoài",
    "BUREAU_ACTIVE_RATIO": "CIC: tỉ lệ khoản vay còn dư nợ trên tổng số khoản vay đã ghi nhận",
    "BUREAU_COUNT": "CIC: số khoản vay ở tổ chức tín dụng khác",
    "PREV_COUNT": "số đơn vay trước ở Home Credit",
    "PREV_REFUSED_RATIO": "tỉ lệ đơn vay trước bị từ chối trên tổng số đơn",
    "PREV_APPROVED_RATIO": "tỉ lệ đơn vay trước được duyệt trên tổng số đơn",
    "INST_LATE_RATIO": "tỉ lệ kỳ trả góp bị trễ hạn trên tổng số kỳ",
    "INST_UNDERPAID_RATIO": "tỉ lệ kỳ trả góp khách trả thiếu trên tổng số kỳ",
}

# `APP_CREDIT_RATIO` của previous.py: AMT_APPLICATION / AMT_CREDIT
BASE["APP_CREDIT_RATIO"] = "tỉ lệ số tiền khách xin vay trên số tiền thực được duyệt (>1 = bị cắt bớt)"

def _split(name: str) -> tuple[str | None, str | None, str | None, str] | None:
    """Tách tên thành (bảng, bộ lọc, cột gốc, phép tổng hợp). None nếu không tra được cột gốc."""
    parts = name.split("_")
    table = None
    if parts and parts[0] in TABLE:
        table, parts = parts[0], parts[1:]
    filt = None
    if parts and parts[0] in FILTER:
        filt, parts = parts[0], parts[1:]
    agg = None
    if len(parts) > 1 and parts[-1] in AGG:
        agg, parts = parts[-1], parts[:-1]
    base = "_".join(parts)
    if base not in BASE:
        return None
    return table, filt, base, agg

def describe(name: str) -> str:
    if name in WHOLE_NAME:
        return WHOLE_NAME[name]
    parsed = _split(name)
    if parsed is None:
        return name
    table, filt, base, agg = parsed
    parts = [f"{TABLE[table]}: {BASE[base]}" if table else BASE[base]]
    if filt:
        parts.append(FILTER[filt])
    if agg:
        parts.append(AGG[agg])
    return ", ".join(parts)


def has_description(name: str) -> bool:
    """True nếu tra được nghĩa thật (không rơi về tên thô)."""
    return name in WHOLE_NAME or _split(name) is not None

# Giải mã categorical
# `features/builder.py` label-encode cột categorical rồi lưu bản đồ nhãn -> mã ở `label_maps.json`.
# SHAP nhìn thấy mã (0,1,2...) nên câu giải thích phải dịch ngược, nếu không sẽ in ra "CODE_GENDER = 1".

_CODE_TO_LABEL: dict[str, dict[int, str]] | None = None

def _label_maps() -> dict[str, dict[int, str]]:
    global _CODE_TO_LABEL
    if _CODE_TO_LABEL is None:
        from src.ph1_credit_risk.features.builder import LABEL_MAPS_PATH
        try:
            raw = json.loads(LABEL_MAPS_PATH.read_text())
        except FileNotFoundError:
            raw = {}
        _CODE_TO_LABEL = {col: {int(code): label for label, code in m.items()} for col, m in raw.items()}
    return _CODE_TO_LABEL

def decode_category(feature: str, code: object) -> str | None:
    """Mã label-encoded -> nhãn gốc ('M', 'Higher education'). None nếu không phải cột categorical."""
    m = _label_maps().get(feature)
    if m is None:
        return None
    try:
        return m.get(int(code))
    except (TypeError, ValueError):
        return None
