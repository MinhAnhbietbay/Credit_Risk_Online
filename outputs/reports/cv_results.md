# CV 5-fold — phần CV 80% (246008 dòng, 80 feature)

Holdout holdout_ids.json không chạm. `auc_mean ± auc_std` qua fold; `oof_auc` trên toàn OOF.

```
              model  auc_mean  auc_std  oof_auc  fit_seconds
           catboost    0.7803   0.0008   0.7802     119.4289
           lightgbm    0.7797   0.0010   0.7795      14.0157
            xgboost    0.7784   0.0006   0.7783      27.9812
logistic_regression    0.7665   0.0015   0.7665      45.9587
      random_forest    0.7617   0.0015   0.7617      96.0961
```

Cập nhật bởi `scripts/ph1_train.py --models all` lúc 2026-09-24 04:49.
