# Execution Playbook

This document defines how to run the harness in real operations and experiments.

## 1. Pre-checks
- Activate `venv`
- Verify data files exist
- Verify write permission for model output directories

## 2. Single-Agent Debug Run

```bash
python main.py solar_forecast
python main.py wind_forecast
python main.py consumption_forecast
```

Or use index-based execution:

```bash
python main.py 1
python main.py 2
python main.py 5
```

## 3. Full Orchestration Run

```bash
python main.py all
```

## 4. Runtime Checkpoints
- Training return code for each attempt
- Result JSON creation
- Best-model selection
- Validity of deployed file paths
- Final prediction artifacts generated

## 5. Report Verification
Check the latest reports in project root.

```bash
ls -t *_training_report_*.log | head -n 5
```

Required report sections:
- `[rounds]`
- `[best]`
- `[deployed]`

## 6. Troubleshooting Guide
- All attempts failed: verify data paths, dependencies, and timeout settings
- Abnormal score: verify metric direction contract (`min`/`max`)
- Inference fails after deployment: verify `model_path`/`feature_path` consistency

## 7. Reproducibility Procedure (for Papers)
1. Freeze the same data snapshot
2. Use the same candidate sets and thresholds
3. Preserve original report files
4. Cite both score and params when creating result tables
