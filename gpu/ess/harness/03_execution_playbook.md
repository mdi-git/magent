# Harness Execution Playbook

`do_harness2.py`의 최적화 실행 계획 및 가이드입니다.

## Step 1: Data Preparation
`gpu/ess/data/total.csv` 파일을 입력으로 받아 각 시간(hour)별 태양광 발전량(solar), 풍력 발전량(wind), 소비전력(powermeter)을 로드합니다. 여기에 TOU 가격표(`get_price` 함수)를 결합하여 일별/시간별 데이터를 시계열 그룹핑합니다.

## Step 2: Agent Simulation (Cost & Carbon)
그리드 서치 파라미터가 부여되면 시뮬레이션 환경 내에서 `soc_cost`, `soc_carb` 두 상태를 별도로 추적하며 에이전트 정책을 적용합니다. 
1. TOU 가격이 Peak 기준선을 초과하면 강방전, Off-peak 기준선 이하이면 집중 충전을 수행합니다.
2. 잉여 재생에너지 (solar+wind)가 기준선을 초과하면 신재생 에너지 저장을 우선시하여 탄소 절감을 확보하고, 야간 기준선을 활용해 방전을 스케줄링합니다.

## Step 3: Score Function & Comparison
- 가중 스코어 산출: `Cost / 10000 + Carbon / 0.01` 과 같은 스코어 보정치를 통해 비용과 탄소 간의 밸런스를 측정.
- Iteration 순회 후, 베이스라인 스코어(GA Cost: 260,483, GA Carbon: 0.095620)와 대비하여 `Cost Gain %` 및 `Carbon Gain %`를 표출합니다.
- (업데이트) 그리드 탐색의 정밀도를 올리고 휴리스틱 룰을 보정하면 GA 비용도 추월할 수 있도록 `MAX_ACTION` 및 방전 룰셋을 고도화할 수 있습니다.
