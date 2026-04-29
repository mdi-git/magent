# Search Policy (Adaptive Tuning)

## 정책 목표
- 고정 3회 반복이 아닌 적응형 탐색
- 계산 비용 제어(최대 4회)
- 충분히 좋은 점수에서 조기 종료

## 공통 정책
- `MAX_TRIES = 4`
- `MIN_TRIES_BEFORE_EARLY_STOP = 2`
- 첫 2회는 반드시 수행
- 이후 best score가 임계치 도달 시 즉시 종료

## 도메인별 예시 임계치
- Solar (`rmse`, min): `<= 20.0`
- Wind (`r2`, max): `>= 0.85`
- Consumption (`mae`, min): `<= 1200.0`

> 임계치는 초기값이며, 운영 데이터 분포/비즈니스 요구에 맞게 재조정해야 한다.

## 후보군 설계 원칙
1. Exploit 후보: 기존 검증된 파라미터 근처
2. Explore 후보: 모델 복잡도/regularization 변화를 포함
3. 비용 고려: 너무 무거운 조합은 후순위

## 예시 후보군 (Wind)

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

## 선택 로직 의사코드

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
