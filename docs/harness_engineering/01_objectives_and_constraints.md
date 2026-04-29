# Objectives and Constraints

## 1. System Objective
Automate the following process for the three forecast agents (solar, wind, consumption).

1. Run the training script
2. Search over hyperparameter candidates
3. Select the best model based on metric performance
4. Deploy the best model to the inference path
5. Execute prediction and return results/logs

## 2. Optimization Objective
- Solar: minimize RMSE
- Wind: maximize R2
- Consumption: minimize MAE

## 3. Iteration Constraints
- Maximum attempts: 4 (`<5`)
- Minimum attempts: 2
- Early stop: terminate immediately when the acceptable threshold is reached

## 4. Operational Constraints
- Python interpreter: prioritize project `venv/bin/python`
- MCP timeout: set large enough to cover full training + inference
- Failure tolerance: some attempts may fail, but best-model selection must still be possible
- Logging: write timestamped reports in project root (`YYYYMMDD_HHMM`)

## 5. Reproducibility Requirements
Record the following for every attempt.

- Input hyperparameters
- Training/evaluation return code
- Metric name and score
- Selection decision
- Deployed model path

## 6. Artifacts
- Agent run result JSON
- `{agent}_training_report_YYYYMMDD_HHMM.log`
- Model files and feature/scaler files (agent-specific)
