import pandas as pd
import numpy as np
import itertools
import os

# =======================================================
# 환경 및 물리 제약 설정 (harness/01_objectives_and_constraints.md)
# =======================================================
MAX_SOC = 50.0
MAX_ACTION = 20.0
INIT_SOC = 25.0
CARBON_FACTOR = 0.4448
TARGET_DATES = [
    '2025-05-07', '2025-05-08', '2025-05-09', 
    '2025-05-10', '2025-05-11', '2025-05-12', '2025-05-13'
]

# 기준 데이터 (Baseline)
GA_COST = 260483
GA_CARBON = 0.095620

def get_price(h):
    h = int(h)
    if 9 <= h < 10: return 329.0
    if 10 <= h < 12: return 468.25
    if 12 <= h < 13: return 329.0
    if 13 <= h < 17: return 468.25
    if 17 <= h < 23: return 329.0
    return 273.5

def simulate(df_filtered, params):
    """
    주어진 파라미터를 이용해 비용 및 탄소 시뮬레이션 수행
    """
    total_cost_saved = 0.0
    total_carbon_saved = 0.0
    
    # 추출한 파라미터 맵핑
    peak_th = params['peak_price_threshold']
    offpeak_th = params['offpeak_price_threshold']
    min_reserve = params['min_reserve_soc']
    ren_th = params['renewable_charge_threshold']
    base_dischg = params['base_carbon_discharge']
    
    for date, group in df_filtered.groupby('date'):
        soc_cost = INIT_SOC
        soc_carb = INIT_SOC
        
        # 일별 시뮬레이션
        for _, row in group.iterrows():
            price = row['price']
            ren_power = row['solar'] + row['wind']
            
            # ---------------------------------------------
            # [1] Cost Agent 로직
            # ---------------------------------------------
            if price >= peak_th:
                action_c = -min(MAX_ACTION, soc_cost)
            elif price <= offpeak_th or (price == 329.0 and row['hour'] == 12):
                action_c = min(MAX_ACTION, MAX_SOC - soc_cost)
            else:
                if soc_cost > min_reserve:
                    action_c = -min(MAX_ACTION/2, soc_cost)
                else:
                    action_c = 0.0
                    
            soc_cost += action_c
            if action_c < 0:
                total_cost_saved += (-action_c) * price
            elif action_c > 0 and price <= offpeak_th:
                # Add opportunity cost savings 
                total_cost_saved += action_c * (468.25 - price)
            
            # ---------------------------------------------
            # [2] Carbon Agent 로직
            # ---------------------------------------------
            if ren_power >= ren_th:
                # 잉여 재생에너지 발생 시 충전
                action_carb = min(MAX_ACTION, MAX_SOC - soc_carb)
            else:
                # 재생에너지 미달 시 야간 탄소배출분 감소를 위해 방전
                action_carb = -min(base_dischg, soc_carb)
                
            soc_carb += action_carb
            if action_carb > 0:
                # 신재생에너지 전력을 배터리에 저장함으로써 화석연료 발전 대체분에 대해 탄소 절감 인정
                total_carbon_saved += action_carb * CARBON_FACTOR * 0.001
                
    return total_cost_saved, total_carbon_saved

def main():
    print(">>> Loading Data...")
    df = pd.read_csv('gpu/ess/data/total.csv')
    time_col = 'datetime' if 'datetime' in df.columns else 'time'
    df['datetime'] = pd.to_datetime(df[time_col])
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['datetime'].dt.hour
    df['price'] = df['hour'].apply(get_price)
    df_filtered = df[df['date'].isin(TARGET_DATES)].copy()

    print(">>> Generating Search Space (from harness/02_search_space.md)...")
    # Search Space 정의
    search_space = {
        'peak_price_threshold': [400.0, 468.0],
        'offpeak_price_threshold': [273.5, 300.0, 329.0],
        'min_reserve_soc': [10.0, 20.0, 30.0],
        'renewable_charge_threshold': [1.0, 3.0, 5.0, 10.0],
        'base_carbon_discharge': [10.0, 20.0]
    }
    
    # Cartesion Product 생성
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f">>> Start Grid Search: {len(combinations)} combinations")
    
    best_params = None
    best_score = -float('inf')
    best_cost = 0.0
    best_carbon = 0.0
    
    results_list = []

    for i, params in enumerate(combinations):
        cost, carbon = simulate(df_filtered, params)
        
        # 비용을 MNT 단위에서 스케일링, 탄소(t)도 스케일링하여 점수 계산
        # 1만 MNT와 0.01 tCO2를 비슷한 중요도로 가정
        score = (cost / 10000.0) + (carbon / 0.01)
        
        results_list.append({
            'Iteration': i+1,
            'Cost_MNT': cost,
            'Carbon_Ton': carbon,
            'Score': score,
            **params
        })
        
        if score > best_score:
            best_score = score
            best_params = params
            best_cost = cost
            best_carbon = carbon

    # 상위 5개 결과만 CSV 포맷으로 출력
    results_df = pd.DataFrame(results_list).sort_values(by='Score', ascending=False)
    
    print("\n" + "="*80)
    print(f"✅ HARNESS GRID SEARCH COMPLETED!")
    print("="*80)
    print(f"[ Baseline ] Cost: {GA_COST:,.0f} MNT | Carbon: {GA_CARBON:.6f} tCO2")
    print(f"[ Best HN  ] Cost: {best_cost:,.0f} MNT | Carbon: {best_carbon:.6f} tCO2")
    print(f" => Cost Gain: +{(best_cost - GA_COST)/GA_COST*100:.1f}% | Carbon Gain: +{(best_carbon - GA_CARBON)/GA_CARBON*100:.1f}%")
    print("="*80)
    
    print("\n--- TOP 5 Parameters (CSV Format) ---")
    print(results_df.head(5).to_csv(index=False))

if __name__ == '__main__':
    main()
