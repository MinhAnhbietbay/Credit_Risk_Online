# Credit Risk Analytics System — PH1

Dự đoán xác suất vỡ nợ trên bộ dữ liệu **Home Credit Default Risk**,
gồm 7 bảng dữ liệu quan hệ (application, bureau, previous_application, POS_CASH, credit_card,
installments...). 

PH1 là quy trình phân tích/mô hình hoá offline

## Cài đặt

```bash
python -m venv credit_risk
source credit_risk/bin/activate
pip install -r requirements.txt
```

Tải `data/home-credit-default-risk.zip` (dataset gốc từ Kaggle) vào thư mục `data/` trước khi chạy bước 1.

## Cấu trúc thư mục

```
src/
  config.py                        # đường dẫn, seed, hằng số toàn cục
  ph1_credit_risk/
    data/            loader.py, schema.py       — đọc & validate 7 bảng CSV gốc
    features/        application.py, bureau.py, previous.py, pos_cash.py,
                      credit_card.py, installments.py, builder.py, selection.py
                      — sinh feature từng bảng, gộp lại, lọc/chọn feature
    modeling/         split.py, registry.py, cv.py, ensemble.py
                      — chia CV/holdout, định nghĩa 5 model, chạy CV, trộn ensemble
    evaluation/       metrics.py
                      — precision/recall/f1/KS/Brier tại ngưỡng, dò ngưỡng theo flag-rate/Youden
    explain/          shap_explain.py, glossary.py
                      — SHAP out-of-fold cho model cây + từ điển mô tả feature tiếng Việt
    stress/           macro_scenarios.py
                      — kịch bản tăng gánh nặng trả nợ (hệ số đo từ dữ liệu, không gõ tay)
scripts/                            # các CLI chạy pipeline
outputs/
  models/    outputs/models/ph1_<model>_fold<k>.joblib, ph1_ensemble.joblib
  oof/       out-of-fold prediction từng model + ensemble (outputs/oof/)
  reports/   báo cáo .md/.csv/.json chính thức
  plots/     ROC, calibration, SHAP, stress bands...
  evidence/  bằng chứng cho từng quyết định thiết kế không phải số liệu final
```

## Pipeline — chạy theo đúng thứ tự

| # | Script | Việc gì | Output chính |
|---|---|---|---|
| 1 | `ph1_unzip_data.py` | Giải nén dữ liệu gốc | `data/home-credit-default-risk/*.csv` |
| 2 | `ph1_build_features.py` | Sinh & gộp feature từ 7 bảng | `data/processed/features_train\|test.parquet` |
| 3 | `ph1_select_features.py` | Lọc tương quan/missing/hằng số, chọn top-80 theo LightGBM gain | `data/processed/selected_features.json` |
| 4 | `ph1_train.py --models all` | CV 5-fold cho 5 model (LR, RF, XGBoost, LightGBM, CatBoost) | `outputs/oof/*_oof.npy`, `outputs/models/ph1_*_fold*.joblib`, `outputs/reports/cv_results.md` |
| 5 | `ph1_ensemble.py` | So sánh rank-average vs stacking trên OOF, chọn ensemble cuối | `outputs/models/ph1_ensemble.joblib`, `outputs/reports/ensemble.md` |
| 6 | `ph1_evidence_threshold_basic.py` | Xác định ngưỡng phân loại từ tỉ lệ từ chối lịch sử của Home Credit (không bịa) | `outputs/reports/threshold_basis.json` |
| 7 | `ph1_report.py` | **Chấm holdout đúng một lần** — bảng so sánh 5 model + ensemble, ROC/calibration | `outputs/reports/model_comparison.csv`, `holdout_metrics.json` |
| 8 | `ph1_shap.py [--sample 5000]` | SHAP out-of-fold cho CatBoost (model cây tốt nhất) | `outputs/reports/shap.md`, `shap_global.csv` |
| 9 | `ph1_stress.py [--rebuild-scenarios]` | Stress-test gánh nặng trả nợ toàn danh mục bằng model cuối | `outputs/reports/stress_test.md` |

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
```

## Kết quả chính

**So sánh model (AUC OOF, 5-fold CV trên 246,008 hồ sơ, 80 feature):**

| Model | AUC mean ± std |
|---|---|
| CatBoost | 0.7803 ± 0.0008 |
| LightGBM | 0.7797 ± 0.0010 |
| XGBoost | 0.7784 ± 0.0006 |
| Logistic Regression | 0.7665 ± 0.0015 |
| Random Forest | 0.7617 ± 0.0015 |
| **Ensemble (stacking 5 model)** | **0.7834** |

**Model cuối:** stacking (LogisticRegression trên logit của OOF 5 model, CV lồng để không rò rỉ).

**Holdout (chấm một lần, 61,503 hồ sơ):** AUC **0.7874**, KS 0.4371, Brier 0.0658 — tại ngưỡng 0.1173
(tương ứng tỉ lệ từ chối lịch sử ~21.9% của Home Credit): Precision 0.224, Recall 0.599, F1 0.326.

**SHAP (CatBoost, out-of-fold):** `EXT_SOURCE_MEAN` chi phối rõ rệt, theo sau là `DAYS_EMPLOYED`,
`PREV_DAYS_LAST_DUE_1ST_VERSION_MAX`, `CREDIT_TERM`, `NAME_EDUCATION_TYPE`. Chi tiết: `outputs/reports/shap.md`.

**Stress-test:** cú sốc *chi phí trả nợ* (annuity/credit tăng theo phân vị p75/p90 của danh mục) ảnh hưởng
nhẹ (PD trung bình đổi ±0.5–0.8 điểm %); cú sốc *hành vi trả nợ xấu đi* ảnh hưởng mạnh hơn nhiều (tỉ lệ
bị chặn tăng từ 21.7% lên tới 32.4%). Đây là **phân tích độ nhạy, không phải dự báo nhân quả** — xem giới
hạn đầy đủ ở `outputs/reports/stress_test.md`.

## Giới hạn đã biết

- `EXT_SOURCE_*` là điểm hộp đen từ nguồn ngoài — không rõ cách tính, không kiểm chứng được có thiên lệch hay không.
- SHAP giải thích **CatBoost**, không phải trực tiếp ensemble stacking dùng để ra quyết định cuối.
- Stress-test không mô phỏng hành vi khách hàng thực tế (vay thêm nơi khác, cơ cấu nợ, bỏ nợ).
- `CODE_GENDER` bị loại chủ động khỏi feature set (thuộc tính được bảo vệ).

## Test
```bash
pytest
```
