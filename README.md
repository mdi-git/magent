# magent

`magent` is a multi-agent forecasting and anomaly-detection project for microgrid operations.
The core flow is **(for forecast agents) training -> best model selection -> prediction -> report logging**.

## Overview

- The orchestrator runs multiple MCP agents in sequence.
- Forecast agents (`solar`, `wind`, `consumption`) run training as part of execution.
- Hyperparameter candidates are evaluated, and the best model is selected by score.
- Each training round and final selection is recorded in timestamped logs.

## Project Structure (Core)

```text
magent/
  main.py
  magent_agents/
    microgrid_balance_orchestrator.py
    solar_forecast_mcp.py
    wind_forecast_mcp.py
    consumption_forecast_mcp.py
    solar_anomaly_mcp.py
    wind_anomaly_mcp.py
  solar_train/
  wind_predict_train/
  powermeter_train/
  gpu/
  docs/harness_engineering/
```

## Quick Start

### 1) Environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a single agent

```bash
python main.py solar_forecast
python main.py wind_forecast
python main.py consumption_forecast
```

You can also run by index.

```bash
python main.py 1
python main.py 2
python main.py 5
```

### 3) Run full orchestration

```bash
python main.py all
```

## Check logs/results

- Training reports are generated in the project root (`*_training_report_YYYYMMDD_HHMMSS.log`).
- Check recent reports:

```bash
ls -t *_training_report_*.log | head -n 5
```

## Harness Docs

For the training/evaluation contract and execution policy, refer to:

- `docs/harness_engineering/00_overview.md`
- `docs/harness_engineering/02_train_eval_contract.md`
- `docs/harness_engineering/03_search_policy.md`
- `docs/harness_engineering/04_execution_playbook.md`

## License

This project is licensed under the MIT License. See `LICENSE` for details.
