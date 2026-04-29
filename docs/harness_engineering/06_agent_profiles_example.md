# Agent Profiles Example

This file is an example of managing per-agent objectives, metrics, early-stop criteria, and search candidates in one place.

```yaml
agents:
  solar_forecast:
    metric:
      name: rmse
      direction: min
      acceptable: 20.0
    max_tries: 4
    min_tries_before_early_stop: 2
    candidates:
      - learning_rate: 0.05
        cat_depth: 4
        lgbm_n_estimators: 150
        xgb_n_estimators: 200
        xgb_max_depth: 4
      - learning_rate: 0.10
        cat_depth: 6
        lgbm_n_estimators: 200
        xgb_n_estimators: 300
        xgb_max_depth: 6
      - learning_rate: 0.20
        cat_depth: 8
        lgbm_n_estimators: 250
        xgb_n_estimators: 400
        xgb_max_depth: 8
      - learning_rate: 0.08
        cat_depth: 5
        lgbm_n_estimators: 300
        xgb_n_estimators: 500
        xgb_max_depth: 5

  wind_forecast:
    metric:
      name: r2
      direction: max
      acceptable: 0.85
    max_tries: 4
    min_tries_before_early_stop: 2
    candidates:
      - learning_rate: 0.02
        iterations: 60000
        depth: 3
        l2_leaf_reg: 3
      - learning_rate: 0.04
        iterations: 100000
        depth: 2
        l2_leaf_reg: 3
      - learning_rate: 0.08
        iterations: 80000
        depth: 4
        l2_leaf_reg: 5
      - learning_rate: 0.03
        iterations: 120000
        depth: 5
        l2_leaf_reg: 7

  consumption_forecast:
    metric:
      name: mae
      direction: min
      acceptable: 1200.0
    max_tries: 4
    min_tries_before_early_stop: 2
    candidates:
      - learning_rate: 0.00001
        n_estimators: 8000
        num_leaves: 31
        max_depth: -1
      - learning_rate: 0.00005
        n_estimators: 10000
        num_leaves: 63
        max_depth: -1
      - learning_rate: 0.0001
        n_estimators: 12000
        num_leaves: 127
        max_depth: 8
      - learning_rate: 0.00003
        n_estimators: 15000
        num_leaves: 63
        max_depth: 10
```

## How to Use
- Use this profile as source-of-truth data instead of hardcoding values
- Operators can change policy by editing only `acceptable` and candidate sets
- For papers/reports, attach this file as the experiment configuration reference
