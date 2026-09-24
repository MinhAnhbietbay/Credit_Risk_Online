# Task 9 — Cách trộn ensemble

**Quyết định:** `stacking_all5` (AUC OOF 0.7834).

**So sánh:** rank-average (5 model đều / 5 model trọng số AUC / 3 boosting) và stacking LR (5 / 3 boosting), tất cả chấm trên cùng OOF phần CV.

**Số liệu:** model đơn tốt nhất `catboost` 0.7802; cấu hình chọn hơn +0.0031.

```
                    config  n_members      auc
             stacking_all5          5 0.783384
           stacking_boost3          3 0.782982
     rank_avg_boost3_equal          3 0.782889
rank_avg_all5_auc_weighted          5 0.780823
       rank_avg_all5_equal          5 0.780669
                  catboost          1 0.780243
                  lightgbm          1 0.779545
                   xgboost          1 0.778280
       logistic_regression          1 0.766453
             random_forest          1 0.761671
```
