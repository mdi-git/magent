# Objectives and Constraints

## 1. Primary Objectives
하네스(Harness Multi-agent)의 ESS 제어 알고리즘 최적화를 수행하여, 유전 알고리즘(GA) 기반 결과(`ess_cost.csv`, `ess_ga_carbon.csv`) 대비 향상된 성능을 도출한다.

- **목표 1 (Cost Reduction)**: TOU(계시별 요금제) 단가를 기반으로 가장 비싼 시간대(Peak)에 방전하고 저렴한 시간대(Off-peak)에 충전하여 전력망 구매 비용을 최소화(또는 수익 최대화)한다.
- **목표 2 (Carbon Reduction)**: 태양광(`solar`)과 풍력(`wind`) 발전량이 많은 시간대에 집중적으로 충전하고, 발전량이 없는 야간/새벽 시간대에 방전하여 화석연료 유래 전력 사용을 최소화한다.

## 2. Constraints
- **ESS 물리적 한계**:
  - 최대 용량 (Max SOC): 50.0 kWh
  - 충/방전 최대 속도 (Max Action): 20.0 kW/h
  - 초기 충전량 (Init SOC): 25.0 kWh
- **제어 로직 제약**:
  - 충전 시, 전체 ESS 용량을 초과할 수 없음.
  - 방전 시, 현재 남아있는 잔여 배터리 용량(SOC) 미만으로만 방전 가능.

## 3. Baseline Metrics
- **Cost (MNT)**: GA 베이스라인 기준 총 절감액 (약 214,925 MNT)을 초과할 것.
- **Carbon (tCO2-eq)**: GA 베이스라인 기준 총 절감량 (약 0.095620 tCO2-eq)을 초과할 것.
