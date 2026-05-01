# Hyperparameter Search Space

하네스 스케줄링 시뮬레이터(`do_harness2.py`)가 튜닝해야 할 파라미터와 범위를 정의합니다.

## 1. Cost Optimization (비용절감) Parameter
- **peak_price_threshold**: 방전을 시작할 기준 TOU 가격 (Peak)
  - `[400.0, 468.25]` (기본 Peak는 468.25)
- **offpeak_price_threshold**: 충전을 시작할 기준 TOU 가격 (Off-peak)
  - `[273.5, 300.0, 329.0]` (기본 Off-peak는 273.5, Mid-peak는 329.0)
- **min_reserve_soc**: 중간 가격대(Mid-peak)에서 과방전을 방지하기 위한 최소 보존 SOC
  - `[10.0, 20.0, 30.0]`

## 2. Carbon Optimization (탄소절감) Parameter
- **renewable_charge_threshold**: 재생에너지(태양광+풍력) 총 발전량이 얼마 이상일 때 즉각 ESS에 충전을 시작할지 결정하는 임계값 (kW).
  - `[1.0, 3.0, 5.0, 10.0]`
- **base_carbon_discharge**: 야간에 탄소 발전분을 대체하기 위해 방전할 고정 용량 비율(또는 절대량)
  - `[10.0, 20.0]`

## 3. Metric Weights
실험 결과를 하나의 점수(Score)로 판단할 때, 비용(MNT)과 탄소(tCO2-eq)를 환산하여 가중합으로 최적 모델을 선정합니다.
