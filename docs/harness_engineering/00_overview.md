# Harness Engineering Overview for `magent`

## Purpose
This document defines a Harness Engineering design to achieve the core `magent` pipeline with **minimal code changes and maximum operational reproducibility**.

- Automatically trigger training when forecast agents run
- Perform repeated training with hyperparameter tuning
- Select the best model by score
- Deploy the selected model to the inference path and run prediction
- Record process and outcomes in reports

## Why Harness Engineering
Feature implementation alone leaves the following issues:

1. Experiments are executed differently by each operator.
2. Stop conditions and search policy are hardcoded, making changes costly.
3. Inconsistent report formats make post-analysis difficult.
4. It is hard to run operations, papers, and reproducibility tests with one workflow.

Harness Engineering addresses these issues by:

- Defining execution rules as explicit contracts
- Separating search policy into document-driven parameters
- Standardizing evaluation and stopping criteria
- Structuring output reports

## Recommended Directory

```text
docs/harness_engineering/
  00_overview.md
  01_objectives_and_constraints.md
  02_train_eval_contract.md
  03_search_policy.md
  04_execution_playbook.md
  05_report_template.md
  06_agent_profiles_example.md
```

## Scope
- `solar_forecast_agent`
- `wind_forecast_agent`
- `consumption_forecast_agent`

Anomaly agents (`solar_anomaly`, `wind_anomaly`) are outside the first rollout scope of this harness, but the design supports extension with the same template.
