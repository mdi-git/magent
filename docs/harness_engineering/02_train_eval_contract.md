# Train/Eval Contract (MD Spec)

This document defines the input/output contract between training scripts and the Harness (orchestration layer).

## 1. Input Contract (Environment Variables)

Required:
- `MAGENT_TRAIN_RESULT_PATH`: output path for training result JSON

Optional (Common):
- `MAGENT_LEARNING_RATE`

Optional (Solar):
- `MAGENT_CAT_DEPTH`
- `MAGENT_LGBM_N_ESTIMATORS`
- `MAGENT_XGB_N_ESTIMATORS`
- `MAGENT_XGB_MAX_DEPTH`

Optional (Wind):
- `MAGENT_ITERATIONS`
- `MAGENT_DEPTH`
- `MAGENT_L2_LEAF_REG`

Optional (Consumption):
- `MAGENT_N_ESTIMATORS`
- `MAGENT_NUM_LEAVES`
- `MAGENT_MAX_DEPTH`

## 2. Output Contract (Training Result JSON)
The training script must write JSON in the following shape.

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

Required fields:
- `metric_name`
- `score`
- `direction` (`min` or `max`)
- Path fields required for model deployment

## 3. Evaluation Rules (Harness Side)
- `direction=min`: lower score is better
- `direction=max`: higher score is better
- If `returncode != 0` or result JSON parsing fails, that attempt is treated as failed

## 4. Failure Handling
- If no valid result exists across all attempts, the agent run fails
- A report file must still be generated even on failure
