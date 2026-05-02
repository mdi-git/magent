import os
import pandas as pd
import numpy as np
import torch

# ========== 1. 경로 및 환율 설정 ==========
BASE_DIR = r'D:\workspace\논문\ess'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'total.csv')
WEIGHT_PATH = os.path.join(BASE_DIR, 'cost_weight', 'ga', 'ga_best_individual_cost.pt')
SAVE_DIR = os.path.join(BASE_DIR, 'cost')

MNT_PER_KRW = 2.5    
MNT_PER_USD = 3450.0 

# ESS 사양
ESS_CAPACITY = 100.0   
MAX_CHARGE = 20.0      
EFFICIENCY = 0.95
INITIAL_SOC = 0.5

# ========== 2. 시간대별 요금 엔진 (MNT 고정) ==========
def get_mnt_price(dt):
    hour = dt.hour
    if 23 <= hour or hour < 9:
        price_krw = 109.4 # 경부하
    elif (10 <= hour < 12) or (13 <= hour < 17):
        price_krw = 187.3 # 최대부하
    else:
        price_krw = 131.6 # 중간부하
    return price_krw * MNT_PER_KRW

# ========== 3. 데이터 로드 및 가중치 로드(에러 방지) ==========
df = pd.read_csv(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])

if os.path.exists(WEIGHT_PATH):
    loaded_data = torch.load(WEIGHT_PATH, map_location='cpu')
    if isinstance(loaded_data, dict):
        best_actions = np.array(list(loaded_data.values())[0])
    else:
        best_actions = loaded_data.detach().cpu().numpy() if torch.is_tensor(loaded_data) else np.array(loaded_data)
    
    best_actions = best_actions.flatten()
    if len(best_actions) < len(df):
        best_actions = np.concatenate([best_actions, np.zeros(len(df) - len(best_actions))])
    best_actions = best_actions[:len(df)]
else:
    print("⚠️ 가중치 파일 없음 - 랜덤 액션으로 진행")
    best_actions = np.random.uniform(-1, 1, len(df))

# ========== 4. ESS 시뮬레이션 ==========
soc = INITIAL_SOC * ESS_CAPACITY
results = []

for i in range(len(df)):
    current_time = df.loc[i, 'time']
    price_mnt = get_mnt_price(current_time)
    action = best_actions[i]
    energy_change = action * MAX_CHARGE
    
    if energy_change > 0: 
        actual_change = min(energy_change, ESS_CAPACITY - soc, MAX_CHARGE)
        soc += actual_change * EFFICIENCY
        hourly_saving_mnt = 0 
    else:
        actual_change = max(energy_change, -soc, -MAX_CHARGE)
        soc += actual_change / EFFICIENCY
        hourly_saving_mnt = abs(actual_change) * price_mnt
        
    results.append({'Hourly_Saving_MNT': hourly_saving_mnt, 'ESS_Action': actual_change})

final_df = pd.concat([df, pd.DataFrame(results)], axis=1)
final_df['date'] = final_df['time'].dt.date

# ========== 5. 결과 출력 및 저장 ==========
daily_cost = final_df.groupby('date')['Hourly_Saving_MNT'].sum()

print("\n" + "="*65)
print(f"{'날짜':<12} | {'절감액 (MNT)':>15} | {'USD 환산':>10}")
print("-" * 65)
for date, val in daily_cost.items():
    print(f"{str(date):<12} | {val:15,.0f} | {val/MNT_PER_USD:10.2f}")

print("-" * 65)
total_mnt = daily_cost.sum()
print(f"💰 총 절감액 (MNT): {total_mnt:,.2f} ₮")
print(f"💰 총 절감액 (KRW): {total_mnt/MNT_PER_KRW:,.0f} 원")
print(f"💰 총 절감액 (USD): {total_mnt/MNT_PER_USD:,.2f} $")
print("="*65)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
final_df.drop(columns=['date']).to_csv(os.path.join(SAVE_DIR, 'ess_cost.csv'), index=False, encoding='utf-8-sig')