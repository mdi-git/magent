# Training Report Template

Use the template below as the standard report format.

```text
timestamp=20260427_0715
agent=wind_forecast_agent
status=completed

[rounds]
#1 lr=0.02 returncode=0 metric=r2 score=0.6861 params={"learning_rate":0.02,"iterations":60000,"depth":3,"l2_leaf_reg":3}
#2 lr=0.04 returncode=0 metric=r2 score=0.7101 params={"learning_rate":0.04,"iterations":100000,"depth":2,"l2_leaf_reg":3}
#3 lr=0.08 returncode=0 metric=r2 score=0.6856 params={"learning_rate":0.08,"iterations":80000,"depth":4,"l2_leaf_reg":5}

[best]
learning_rate=0.04 metric=r2 score=0.7101
params={"learning_rate":0.04,"iterations":100000,"depth":2,"l2_leaf_reg":3}

[deployed]
model=/abs/path/wind_model_1h.cbm
feature=/abs/path/wind_features_1h.joblib
```

## Writing Rules
- timestamp: `YYYYMMDD_HHMM`
- Keep score as original float (no post-rounding)
- Record `params` exactly as run input
- Keep `rounds`/`best`/`deployed` structure even on failure

## Failure Example

```text
timestamp=20260427_0721
agent=solar_forecast_agent
status=failed
error=solar training failed for all learning rates

[rounds]
#1 lr=0.05 returncode=1 metric=None score=None params={...}
#2 lr=0.1 returncode=1 metric=None score=None params={...}

[best]
none
```
