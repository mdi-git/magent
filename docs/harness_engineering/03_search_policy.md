# Search Policy (Adaptive Tuning)

## Policy Goals
- Use adaptive search instead of a fixed 3-run loop
- Control computation cost (up to 4 attempts)
- Early stop when score is good enough

## Common Policy
- `MAX_TRIES = 4`
- `MIN_TRIES_BEFORE_EARLY_STOP = 2`
- Always run at least the first 2 attempts
- After that, stop immediately once best score reaches the threshold

## Example Thresholds by Domain
- Solar (`rmse`, min): `<= 20.0`
- Wind (`r2`, max): `>= 0.85`
- Consumption (`mae`, min): `<= 1200.0`

> These are initial thresholds and should be recalibrated for production data distribution and business requirements.

## Candidate Design Principles
1. Exploit candidates: near previously validated parameters
2. Explore candidates: include changes in model complexity/regularization
3. Cost-aware ordering: keep very heavy combinations lower priority

## Example Candidate Set (Wind)

```yaml
wind_candidates:
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
```

## Selection Logic Pseudocode

```text
best = None
success_count = 0
for candidate in candidates[:MAX_TRIES]:
  result = train(candidate)
  if result is valid:
    success_count += 1
    best = better(best, result)
    if success_count >= MIN_TRIES and acceptable(best):
      break
if best is None:
  fail
else:
  deploy(best)
```
