# Credit Risk Analytics System

Dự đoán xác suất vỡ nợ (PD) cho hồ sơ vay tiêu dùng, dựa trên bộ dữ liệu Kaggle
**Home Credit Default Risk** — 7 bảng quan hệ, tổng cộng hơn 58 triệu dòng lịch sử tín dụng.
So sánh 5 model + 1 ensemble, giải thích bằng SHAP, stress-test danh mục, kiểm định (rò rỉ, hiệu chỉnh, công bằng, ổn định phân phối), và một hệ thống trực tuyến chạy trên PostgreSQL với web FastAPI + React.

**Nguồn dữ liệu duy nhất:** bộ Kaggle *Home Credit Default Risk*.

**Web:** backend FastAPI (`src/api/`) + frontend React/Vite/TypeScript (`frontend/`)
---

## 1. Bài toán

Home Credit cho vay tiêu dùng với khách hàng ít hoặc không có lịch sử tín dụng chính thức. Câu hỏi nghiệp vụ: hồ sơ nào sẽ gặp khó khăn trả nợ, đủ sớm để từ chối hoặc chuyển thẩm định thủ công.

`application_train` có **307,511 hồ sơ**, tỉ lệ vỡ nợ **8.07%**. Mất cân bằng ~11:1 — đây là bài toán xếp hạng rủi ro, không phải phân loại cân bằng, nên **AUC và KS là thước đo chính**, không phải accuracy.

Ba câu hỏi dự án trả lời:
1. **Ai rủi ro?** - PD cho từng hồ sơ, từ 80 feature tổng hợp trên toàn bộ lịch sử tín dụng.
2. **Vì sao model chấm như vậy?** - SHAP cho từng hồ sơ.
3. **Danh mục chịu được cú sốc đến đâu?** - stress-test gánh nặng trả nợ (chi phí + hành vi).

---

## 2. Cách chạy

```bash
python -m venv credit_risk
source credit_risk/bin/activate
pip install -r requirements.txt
```

Tải `data/home-credit-default-risk.zip` (dataset gốc từ Kaggle) vào thư mục `data/` trước khi chạy bước 1.

### Pipeline offline — một lệnh

```bash
python scripts/ph1_run_all.py                  # toàn bộ chuỗi, đúng thứ tự phụ thuộc
python scripts/ph1_run_all.py --from report    # dùng lại model đã train
python scripts/ph1_run_all.py --only fairness drift
python scripts/ph1_run_all.py --dry-run        # xem danh sách bước
```

Hoặc từng bước:

| # | Script | Việc | Output chính |
|---|---|---|---|
| 0 | `ph1_unzip_data.py` | Giải nén dữ liệu gốc | `data/home-credit-default-risk/*.csv` |
| 1 | `ph1_build_features.py` | Sinh & gộp feature từ 7 bảng | `data/processed/features_train\|test.parquet` |
| 2 | `ph1_select_features.py` | Lọc tương quan/missing/hằng số, chọn top-80 theo LightGBM gain | `data/processed/selected_features.json` |
| 3 | `ph1_train.py --models all` | CV 5-fold cho 5 model | `outputs/oof/*_oof.npy`, `outputs/models/ph1_*_fold*.joblib`, `cv_results.md` |
| 4 | `ph1_ensemble.py` | So sánh rank-average vs stacking trên OOF, chọn ensemble cuối | `outputs/models/ph1_ensemble.joblib`, `ensemble.md` |
| 5 | `ph1_evidence_threshold_basic.py` | Ngưỡng phân loại từ tỉ lệ từ chối lịch sử | `threshold_basis.json` |
| 6 | `ph1_report.py` | Chấm holdout đúng một lần - bảng so sánh 5 model + ensemble, ROC/calibration | `model_comparison.csv`, `holdout_metrics.json` |
| 7 | `ph1_shap.py [--sample 5000]` | SHAP out-of-fold cho CatBoost | `shap.md`, `shap_global.csv` |
| 8 | `ph1_stress.py [--rebuild-scenarios]` | Stress-test | `stress_test.md` |
| 9 | `ph1_leakage.py` | Kiểm rò rỉ thời gian | `leakage.md` |
| 10 | `ph1_uncertainty.py` | Khoảng tin cậy bootstrap | `uncertainty.md` |
| 11 | `ph1_calibration.py` | So hiệu chỉnh Platt/isotonic | `calibration.md` |
| 12 | `ph1_fairness.py` | Số đo công bằng theo giới tính/tuổi | `fairness.md` |
| 13 | `ph1_drift.py` | PSI | `psi.md` |

```bash
python scripts/ph1_unzip_data.py
python scripts/ph1_build_features.py
python scripts/ph1_select_features.py
python scripts/ph1_train.py --models all
python scripts/ph1_ensemble.py
python scripts/ph1_evidence_threshold_basic.py
python scripts/ph1_report.py
python scripts/ph1_shap.py
python scripts/ph1_stress.py
python scripts/ph1_leakage.py
python scripts/ph1_uncertainty.py
python scripts/ph1_calibration.py
python scripts/ph1_fairness.py
python scripts/ph1_drift.py
```

### Web — FastAPI + React (Docker)

Điều kiện: đã chạy xong pipeline offline, tức là đã có `data/processed/`, `outputs/models/` và `outputs/reports/` (`shap_global.csv`, `model_comparison.csv`).

```bash
# 1. Bật PostgreSQL + API (cổng 8000) + web (cổng 5173)
docker compose -f infra/docker-compose.yml up -d --build

# 2. Nạp hồ sơ vào DB và tạo dữ liệu cho form nhập tay (chỉ cần chạy lần đầu)
docker compose -f infra/docker-compose.yml exec api python scripts/ph1_db_seed.py --reset
docker compose -f infra/docker-compose.yml exec api python scripts/ph1_web_assets.py
```

- Web: http://localhost:5173 · Tài liệu API: http://localhost:8000/docs
- PostgreSQL mở ra máy ở cổng 5433 (user/mật khẩu `postgres`, DB `credit_risk`).

Nguồn gốc 80 feature của model: [docs/ph1-nguon-goc-80-feature.md](docs/ph1-nguon-goc-80-feature.md).

### Test

```bash
pytest
```

---

## 3. Cấu trúc thư mục

```
src/
  config.py                        # đường dẫn, seed, hằng số toàn cục
  ph1_credit_risk/
    data/          loader.py, schema.py       — đọc & validate 7 bảng CSV gốc
    features/      application.py, bureau.py, previous.py, pos_cash.py,
                   credit_card.py, installments.py, builder.py, selection.py
                   — sinh feature từng bảng, gộp lại, lọc/chọn feature
    modeling/      split.py, registry.py, cv.py, ensemble.py, scoring.py
                   — chia CV/holdout, định nghĩa 5 model, chạy CV, trộn ensemble
    evaluation/    metrics.py, leakage.py, bootstrap.py, calibration.py, fairness.py, drift.py
                   — precision/recall/f1/KS/Brier, kiểm rò rỉ, bootstrap CI, hiệu chỉnh, công bằng, PSI
    explain/       shap_explain.py, glossary.py
                   — SHAP out-of-fold cho model cây + từ điển mô tả feature tiếng Việt
    stress/        macro_scenarios.py
                   — kịch bản tăng gánh nặng trả nợ (hệ số đo từ dữ liệu, không gõ tay)
  online/          schema, engine, repository, seed, simulator, feature_set, retrain, models
                   — hệ thống trực tuyến: DB, champion/challenger, train lại
  api/             backend FastAPI — routers/ (predict, retrain, predictions, insights,
                   segmentation, stress, catalog), services/, schemas/
frontend/          frontend React + Vite + TypeScript — 6 tab (frontend/src/pages/)
scripts/           CLI cho từng bước pipeline + ph1_run_all.py, db_seed, web_assets
outputs/
  models/    outputs/models/ph1_<model>_fold<k>.joblib, ph1_ensemble.joblib
  oof/       out-of-fold prediction từng model + ensemble
  reports/   báo cáo .md/.csv/.json chính thức (model_comparison, shap, stress_test, leakage,
             calibration, fairness, psi, uncertainty…)
  plots/     ROC, calibration, SHAP, stress bands...
  evidence/  bằng chứng cho từng quyết định thiết kế không phải số liệu final
infra/             docker-compose.yml (PostgreSQL + API + web)
docs/              nguồn gốc 80 feature
tests/             pytest cho feature engineering
```

---

## 4. Hệ thống trực tuyến (PostgreSQL + FastAPI + React)

Vòng đời: hồ sơ vào DB → chấm PD → nhãn thật về dần → train lại → chọn model đang dùng.

**Lược đồ** (`src/online/schema.py`): `applicants`, `applicant_features` (80 feature dạng
**JSONB** — tránh migrate DDL nếu bộ feature đổi), `model_versions`, `predictions`.

**Sáu tab** (frontend React `frontend/src/pages/`, số liệu lấy từ backend `src/api/routers/`):

| Tab | Nội dung |
|---|---|
| Dự đoán | Chấm hồ sơ có sẵn hoặc nhập tay (form hỏi các feature quan trọng nhất theo SHAP, còn lại điền trung vị tập train) + giải thích SHAP |
| Train lại | Tiết lộ nhãn, nhập kết quả thật, train challenger, kích hoạt model |
| Dự đoán vs thực tế | Ma trận nhầm lẫn, phân bố PD theo kết quả thật |
| Insight nghiệp vụ | Tỉ lệ vỡ nợ thật theo từng yếu tố, sinh từ số đo trên DB |
| Phân khúc rủi ro | Nhóm theo PD, đối chiếu dự đoán với thực tế từng nhóm |
| Stress test | Chạy lại §11 trên mẫu danh mục |

**Champion vs challenger.** Champion = ensemble offline, đã qua holdout (AUC 0.7874). Challenger = LightGBM train lại từ DB khi bấm nút trong tab "Train lại" (`src/online/retrain.py`, cần tối thiểu 2,000 hồ sơ đã biết nhãn và 100 ca vỡ nợ). Challenger không tự lên làm model đang dùng; phải bấm kích hoạt. `ModelVersion.role` phân biệt `champion`/`challenger`,`is_active` đảm bảo đúng một model đang phục vụ dự đoán tại một thời điểm. Holdout không được nạp vào DB.

---

## 5. Dữ liệu — 7 bảng

| Bảng | Khoá | Nội dung |
|---|---|---|
| `application_train` | `SK_ID_CURR` | Hồ sơ đang xin vay + nhãn `TARGET` |
| `bureau` | `SK_ID_BUREAU` → `SK_ID_CURR` | Khoản vay ở tổ chức tín dụng khác |
| `bureau_balance` | `SK_ID_BUREAU` | Số dư từng tháng của khoản vay đó |
| `previous_application` | `SK_ID_PREV` → `SK_ID_CURR` | Đơn vay trước ở Home Credit |
| `POS_CASH_balance` | `SK_ID_PREV` | Số dư trả góp POS / vay tiền mặt theo tháng |
| `installments_payments` | `SK_ID_PREV` | Từng kỳ trả góp: hạn, số phải trả, số thực trả |
| `credit_card_balance` | `SK_ID_PREV` | Sao kê thẻ tín dụng theo tháng |

Hai tầng tổng hợp: bảng con gộp về `SK_ID_PREV`/`SK_ID_BUREAU` trước, rồi gộp tiếp về
`SK_ID_CURR` — một hồ sơ có thể có hàng chục khoản vay cũ, mỗi khoản hàng chục tháng số dư.

---

## 6. Pipeline feature

```
7 bảng thô  →  aggregate 2 tầng  →  758 cột  →  lọc  →  561 cột  →  LightGBM gain  →  80 feature
```

- **Aggregate**: mỗi bảng con sinh `MEAN/MAX/MIN/SUM/STD/COUNT` theo `SK_ID_CURR`, có bản lọc
  theo tập con (`ACTIVE` với CIC, `APPROVED`/`REFUSED` với đơn cũ). Cộng thêm feature dẫn xuất
  tính trên từng dòng trước khi gộp: `DPD`/`DBD`, `PAYMENT_PERC`, `UTILIZATION`.
- **Lọc**: bỏ 173 cột tương quan > 0.95, 21 cột thiếu > 85%, 2 cột hằng số, 1 cột bị cấm
  (`CODE_GENDER`, xem §9).
- **Xếp hạng**: LightGBM gain importance trung bình, **chỉ chạy trên phần CV 80%** — xếp hạng có nhìn nhãn nên không được chạm holdout.

**Kỷ luật đánh giá:** tách **holdout 20% (61,503 hồ sơ) một lần duy nhất** ngay từ đầu. Mọi bước có nhìn nhãn - chọn feature, so sánh model, chọn ensemble, chọn ngưỡng - chỉ dùng phần CV 80% (246,008 hồ sơ). Holdout chỉ chấm cho model cuối, đúng một lần.

---

## 7. So sánh model

Stratified 5-fold trên phần CV, chấm **out-of-fold**. Ngưỡng của mỗi model = mức chặn đúng tỉ lệ **21.9%** hồ sơ điểm cao nhất của chính nó nên P/R/F1 so sánh được giữa các model.

| Model | AUC (OOF, CV) | KS | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Ensemble (stacking 5 model)** | **0.7834** | **0.4275** | 0.2169 | 0.5884 | **0.3170** | **0.0662** |
| CatBoost | 0.7803 ± 0.0008 | 0.4227 | 0.2151 | 0.5833 | 0.3143 | 0.1752 |
| LightGBM | 0.7797 ± 0.0010 | 0.4215 | 0.2148 | 0.5826 | 0.3139 | 0.1698 |
| XGBoost | 0.7784 ± 0.0006 | 0.4190 | 0.2138 | 0.5799 | 0.3124 | 0.1635 |
| Logistic Regression | 0.7665 ± 0.0015 | 0.3992 | 0.2071 | 0.5618 | 0.3027 | 0.1960 |
| Random Forest | 0.7617 ± 0.0015 | 0.3904 | 0.2037 | 0.5525 | 0.2976 | 0.1771 |

**Model cuối: stacking 5 model** (LogisticRegression trên logit của OOF, CV lồng để không rò rỉ) - AUC OOF 0.7834, hơn model đơn tốt nhất (CatBoost 0.7802) **+0.0031**.

**Holdout 61,503 hồ sơ chưa từng chạm — chấm một lần:**

| AUC | KS | Brier | Precision | Recall | F1 | Ngưỡng |
|---:|---:|---:|---:|---:|---:|---:|
| **0.7874** | 0.4371 | 0.0658 | 0.2237 | 0.5988 | 0.3257 | 0.1173 |

Khoảng tin cậy 95% (bootstrap 1000 lần, rút phân tầng): AUC holdout **0.7874 [0.7808, 0.7939]**. Holdout **cao hơn** OOF +0.0041, nằm trong dao động đó — không có dấu hiệu khớp quá mức khi chọn model. Ensemble hơn CatBoost là thật nhưng nhỏ: +0.0031 [0.0025, 0.0038] trên OOF, +0.0014 [0.0005, 0.0023] trên holdout (bootstrap theo cặp).

**Note:** 5 model đơn train với `scale_pos_weight`/`class_weight="balanced"`
nên Brier của chúng cao (0.16–0.20) — xác suất chúng trả về là **điểm xếp hạng chưa hiệu chỉnh**,không phải PD. Chỉ ensemble stacking cho xác suất hiệu chỉnh (Brier 0.066). Xếp hạng (AUC/KS) không bị ảnh hưởng, nhưng đọc số như PD thì sai. Thử thêm Platt/isotonic lên ensemble không cải thiện có ý nghĩa nên giữ nguyên.

---

## 8. Ngưỡng chặn lấy từ dữ liệu

Ngưỡng lấy từ rủi ro có thật trong dữ liệu: Home Credit đã từ chối **21.9%** trong 1,327,459 hồ sơ lịch sử có kết quả (`previous_application`). Ngưỡng = mức chặn đúng tỉ lệ đó → **0.1173**.

Đối chiếu bằng KS/Youden (thống kê): ngưỡng 0.082, chặn 32.5%, P 0.178 / R 0.718.

Phân tích chi phí theo số tiền thật cũng đã làm, nhưng không dùng làm ngưỡng chính: ngưỡng ra dao động 0.29–0.59 tuỳ LGD, mà LGD không có trong dữ liệu (proxy chỉ n=48). Chi tiết:

---

## 9. Model coi trọng điều gì (SHAP)

| # | Feature | \|SHAP\| TB | Ý nghĩa |
|---:|---|---:|---|
| 1 | `EXT_SOURCE_MEAN` | 0.3665 | Trung bình 3 điểm tín dụng nguồn ngoài |
| 2 | `DAYS_EMPLOYED` | 0.1288 | Số ngày kể từ khi vào làm công việc hiện tại |
| 3 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | 0.1188 | Ngày tới kỳ trả cuối theo lịch ký ban đầu của đơn vay cũ |
| 4 | `CREDIT_TERM` | 0.1172 | Tiền trả mỗi kỳ / tổng khoản vay |
| 5 | `NAME_EDUCATION_TYPE` | 0.1101 | Trình độ học vấn |
| 6 | `AMT_ANNUITY` | 0.1061 | Số tiền phải trả mỗi kỳ |
| 7 | `EXT_SOURCE_1` | 0.0953 | Điểm tín dụng nguồn ngoài số 1 |
| 8 | `EXT_SOURCE_MAX` | 0.0831 | Điểm tín dụng nguồn ngoài cao nhất |

Xếp hạng đầy đủ: `outputs/reports/shap_global.csv`. Plot:
`outputs/plots/ph1_shap_{beeswarm,bar,waterfall}.png`.

**SHAP mô tả CatBoost - model cây thành phần tốt nhất, không phải ensemble stacking**  Tính **out-of-fold**: mỗi hồ sơ giải thích bằng đúng model fold không train trên nó.

**Insight nghiệp vụ:**
- `EXT_SOURCE_*` chi phối, lớn hơn phần còn lại một bậc — nhưng là điểm hộp đen từ nguồn ngoài,
  không biết tính từ gì nên không kiểm được có thiên lệch hay không.
- Lịch sử trả nợ ở đơn vay cũ (`PREV_DAYS_LAST_DUE_1ST_VERSION_MAX`, `INST_LATE_RATIO`) đứng ngay
  sau — **hành vi trả nợ quá khứ dự báo tốt hơn thông tin nhân thân**.
- `CREDIT_TERM`/`AMT_ANNUITY` (gánh nặng trả nợ) quan trọng nhưng quan hệ với vỡ nợ **không đơn
  điệu** — xem §11, phát hiện quan trọng nhất của dự án.

---

## 10. Công bằng: đã bỏ `CODE_GENDER`

Ở bản chạy có `CODE_GENDER`, SHAP cho thấy đây là feature quan trọng thứ 4/80. Trong dữ liệu: nam 10.14% vỡ nợ, nữ 7.00% - model cộng thẳng điểm rủi ro cho hồ sơ nam. Giới tính là thuộc tính được bảo vệ; dùng nó để quyết định cấp tín dụng là phân biệt đối xử trực tiếp.

Cái giá: Holdout AUC giảm từ 0.7895 xuống 0.7874 (−0.0021), P/R gần như không đổi.

**Giới hạn :** bỏ tên cột **không** khử được thiên lệch gián tiếp, ngưỡng 0.1173, OOF phần CV:

| | Nữ | Nam | <25 tuổi | 55+ tuổi |
|---|---:|---:|---:|---:|
| Vỡ nợ thật | 7.00% | 10.14% | 12.4% | 5.2% |
| PD trung bình của model | 7.40% | 9.40% | 12.9% | 5.2% |
| Bị chặn | 19.1% | 27.3% | 43.2% | 10.1% |

- 80 feature còn lại **đoán được giới tính với AUC 0.899** - model vẫn có đủ thông tin để phân biệt gián tiếp (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE` tương quan với giới tính và nằm trong top-20 SHAP).
- Theo giới tính: model không phạt nam nặng hơn rủi ro thật của họ (PD trung bình thấp hơn tỉ lệ vỡ nợ thật của nhóm đó). Tỉ số tỉ lệ chặn nữ/nam ≈ 0.70, dưới mốc 0.8 của quy tắc 4/5 (chỉ để tham chiếu)
- Theo tuổi: model hiệu chỉnh đúng trong từng nhóm, nhưng người <25 tuổi bị chặn nhiều gấp ~4.3 lần nhóm 55+. `DAYS_BIRTH` vẫn nằm trong model.

---

## 11. Stress-test

Kịch bản **không** dùng kiểu "lãi suất SBV +2%": lãi suất ngầm của danh mục này là **~48%/năm** (giải ngược từ ~939,000 khoản vay cũ) nên "+2 điểm %" gần như vô hình lên tiền trả mỗi kỳ.

Thay bằng cú sốc đo trên **chính phân vị của danh mục**, hai kênh (gánh nặng trả nợ =
`AMT_ANNUITY / AMT_CREDIT`, đo trên 245,998 hồ sơ phần CV: p50=0.0500, p75=0.0639, p90=0.0950):

| Kịch bản | Kênh | Cú sốc | PD TB | % bị chặn | Chênh |
|---|---|---:|---:|---:|---:|
| `co_so` | — | — | 8.06% | 21.7% | — |
| `chi_phi_bat_loi` | chi phí | ×1.2782 annuity | 8.16% | 22.2% | +0.5đ% |
| `chi_phi_rat_bat_loi` | chi phí | ×1.8996 annuity | 7.80% | 21.0% | **−0.8đ%** |
| `hanh_vi_bat_loi` | hành vi | +25 điểm phân vị | 11.64% | 30.9% | +9.2đ% |
| `hanh_vi_rat_bat_loi` | hành vi | +40 điểm phân vị | 12.07% | 32.4% | +10.7đ% |

**Kịch bản chi phí nặng nhất làm PD giảm** Nhóm `CREDIT_TERM` cao nhất
(gánh nặng/kỳ lớn nhất) lại là vay tiêu dùng POS món nhỏ kỳ hạn rất ngắn - một phân khúc khách hàng ít vỡ nợ hơn hẳn. Model chỉ học tương quan, không có khái niệm nhân quả "trả nặng hơn thì dễ vỡ nợ hơn"; ngoại suy cả danh mục vào vùng đuôi đó không mô phỏng cú sốc mà mô phỏng "giả vờ mọi khách đều là khách vay ngắn hạn". Kênh hành vi (trễ hạn, dùng cạn hạn mức) đơn điệu với rủi ro nên chạy đúng chiều ở cả hai mức.

> Toàn bộ phần này là **phân tích độ nhạy, không phải dự báo**. Không có quan hệ nhân quả nào giữa biến vĩ mô và vỡ nợ được kiểm chứng trên dữ liệu này.

---

## 12. Kiểm định model: rò rỉ, hiệu chỉnh, ổn định phân phối

- **Rò rỉ thời gian**: mọi cột ngày ở bảng phụ tính so với ngày nộp đơn. Cột ghi sự kiện đã xảy ra không có giá trị sau ngày nộp đơn, trừ `bureau.DAYS_CREDIT_UPDATE` (không nằm trong 80 feature). **Không phát hiện rò rỉ.** Giới hạn: không kiểm được `EXT_SOURCE_*` (điểm hộp đen).
- **Hiệu chỉnh xác suất**: Platt/isotonic fit OOF lên ensemble không cải thiện Brier có ý nghĩa — stacking đã tự hiệu chỉnh, giữ nguyên `none`.
- **Ổn định phân phối PSI** ( mốc tham chiếu ngành 0.1/0.25):bphần CV vs holdout: 80/80 feature ổn định (phép tách đúng). Train vs `application_test` (48,744 hồ sơ Kaggle): 78/80 feature ổn định, `CREDIT_INCOME_RATIO` cần theo dõi, `CREDIT_TERM` lệch lớn (PSI ≈ 1.0 — tập test có kỳ hạn vay ngắn hơn). **Điểm model ổn định (PSI 0.0066).**
- **Khoảng tin cậy**: bootstrap 1000 lần
