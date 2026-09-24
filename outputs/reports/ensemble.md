# Ensemble (Task 9)

Chọn theo AUC OOF trên phần CV 80% (246008 dòng); holdout không chạm.

**Model cuối:** `stacking` trên ['catboost', 'lightgbm', 'logistic_regression', 'random_forest', 'xgboost'] — AUC OOF **0.7834** (model đơn tốt nhất `catboost` 0.7802, chênh +0.0031).

```
                    config  n_members    auc
             stacking_all5          5 0.7834
           stacking_boost3          3 0.7830
     rank_avg_boost3_equal          3 0.7829
rank_avg_all5_auc_weighted          5 0.7808
       rank_avg_all5_equal          5 0.7807
                  catboost          1 0.7802
                  lightgbm          1 0.7795
                   xgboost          1 0.7783
       logistic_regression          1 0.7665
             random_forest          1 0.7617
```
