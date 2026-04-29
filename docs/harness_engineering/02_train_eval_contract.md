# Train/Eval Contract (MD Spec)

이 문서는 학습 스크립트와 Harness(오케스트레이션 레이어) 사이의 입출력 계약을 정의한다.

## 1. 입력 계약 (Environment Variables)

필수:
- `MAGENT_TRAIN_RESULT_PATH`: 학습 결과 JSON 저장 경로

공통 선택:
- `MAGENT_LEARNING_RATE`

태양광 선택:
- `MAGENT_CAT_DEPTH`
- `MAGENT_LGBM_N_ESTIMATORS`
- `MAGENT_XGB_N_ESTIMATORS`
- `MAGENT_XGB_MAX_DEPTH`

풍력 선택:
- `MAGENT_ITERATIONS`
- `MAGENT_DEPTH`
- `MAGENT_L2_LEAF_REG`

소비전력 선택:
- `MAGENT_N_ESTIMATORS`
- `MAGENT_NUM_LEAVES`
- `MAGENT_MAX_DEPTH`

## 2. 결과 계약 (Training Result JSON)
학습 스크립트는 아래 JSON을 기록해야 한다.

```json
{
  "learning_rate": 0.04,
  "params": {
    "learning_rate": 0.04,
    "depth": 2,
    "iterations": 100000
  },
  "metric_name": "r2",
  "score": 0.71,
  "direction": "max",
  "model_path": "/abs/path/model.cbm",
  "feature_path": "/abs/path/features.joblib"
}
```

필수 필드:
- `metric_name`
- `score`
- `direction` (`min` or `max`)
- 모델 배포에 필요한 경로 필드

## 3. 평가 규칙 (Harness Side)
- `direction=min`: score 작을수록 우수
- `direction=max`: score 클수록 우수
- `returncode != 0` 또는 결과 JSON 파싱 실패 시 해당 시도는 실패 처리

## 4. 실패 처리
- 전체 시도 중 유효 결과가 하나도 없으면 Agent 실행 실패
- 실패 시에도 리포트 파일은 반드시 생성
