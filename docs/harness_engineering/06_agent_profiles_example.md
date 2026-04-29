# Agent Profiles Example

이 파일은 에이전트별 목표, metric, 조기종료 기준, 탐색 후보군을 한 곳에서 관리하는 예시다.

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

## 사용 방식
- 코드 하드코딩 대신 이 프로파일을 기준 데이터로 사용
- 운영자는 acceptable/후보군만 수정해 정책을 변경
- 논문/보고서에는 이 파일을 실험 설정 근거로 첨부
