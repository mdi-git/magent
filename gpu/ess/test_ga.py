import os
import sys
import torch
import pandas as pd
import numpy as np
from env import ESSEnv

# ==============================================================================
# 1. 설정
# ==============================================================================
TARGET_YEAR = 2025        # 원하는 연도
START_DATE = f'{TARGET_YEAR}-05-01'
END_DATE = f'{TARGET_YEAR}-05-31'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 파일 경로 설정
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'ess_carbon.csv')
WEIGHT_PATH = os.path.join(CURRENT_DIR, 'pt', 'ga_best_individual_0724_1401.pt')
RESULT_DIR = os.path.join(CURRENT_DIR, 'result')
CSV_PATH = os.path.join(RESULT_DIR, 'ga_test_action.csv')

# ==============================================================================
# 2. 함수 정의
# ==============================================================================

def load_data(data_path, start_date, end_date, target_year=None):
    """데이터 로드 및 날짜 필터링 (연도 설정 기능 포함)"""
    if not os.path.exists(data_path):
        print(f"오류: 데이터 파일이 없습니다 -> {data_path}")
        sys.exit()
        
    df = pd.read_csv(data_path)
    
    # 연도 설정 로직
    if target_year is not None:
        # CSV가 MM-DD 형식일 때 앞에 연도 부착
        df['datetime'] = pd.to_datetime(str(target_year) + '-' + df['datetime'], format='%Y-%m-%d')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 날짜 필터링
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
    return df.reset_index(drop=True)

def save_action_csv(dates, actions, amounts, save_path):
    """결과 CSV 저장"""
    df = pd.DataFrame({
        'date': dates,
        'action': actions,  # 1=충전, 0=대기, -1=방전
        'amount_kWh': amounts
    })
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

def print_carbon_saving(total_kwh):
    """탄소 절감량 출력"""
    carbon_factor = 0.4448
    saved = total_kwh * carbon_factor * 0.001
    print(f"-"*30)
    print(f"총 절감된 탄소량: {saved:.2f} tCO2-eq")
    print(f"-"*30)

# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================

if __name__ == "__main__":
    # 폴더 생성
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if not os.path.exists(WEIGHT_PATH):
        print(f"오류: 가중치 파일이 없습니다 -> {WEIGHT_PATH}")
        sys.exit()

    print(f">>> 시뮬레이션 시작: {START_DATE} ~ {END_DATE}")

    # 1. 데이터 로드 (이제 같은 파일 안에 함수가 있으므로 에러 안 남)
    df = load_data(DATA_PATH, START_DATE, END_DATE, target_year=TARGET_YEAR)
    N_DAYS = len(df)
    
    if N_DAYS == 0:
        print("!!! 오류: 해당 기간의 데이터가 없습니다. 날짜를 확인해주세요.")
        sys.exit()
        
    env = ESSEnv(df)

    # 2. 모델(가중치) 로드
    print(f">>> 모델 로드 중: {os.path.basename(WEIGHT_PATH)}")
    try:
        data = torch.load(WEIGHT_PATH, weights_only=False)
        actions = data['actions']
        amounts = data['amounts']
    except Exception as e:
        print(f"!!! 모델 로드 실패: {e}")
        sys.exit()

    # 3. 루프 실행
    dates = []
    actions_list = []
    amounts_list = []
    total_charge_kwh = 0
    state = env.reset()

  
    for i in range(N_DAYS):
        if i >= len(actions):
            
            break

        action = int(actions[i])
        amount = float(amounts[i])
        
        # 날짜 포맷 (MM-dd)
        dates.append(df.iloc[i]['datetime'].strftime('%m-%d'))
        actions_list.append(action)
        amounts_list.append(amount)
        
        if action == 1:
            total_charge_kwh += amount
            
        state, _, done, _ = env.step(action, amount)
        if done:
            break

    # 4. 결과 저장
    save_action_csv(dates, actions_list, amounts_list, CSV_PATH)
    

    # 5. 최종 출력
    print_carbon_saving(total_charge_kwh)