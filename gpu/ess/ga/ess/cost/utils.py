import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np

# 한글 폰트 설정 (Windows)
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

def load_data(data_path, start_date, end_date):
    """
    데이터 로드 및 날짜 필터링
    """
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime("2025-" + df['datetime'], format='%Y-%m-%d')
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    return df.reset_index(drop=True)

def save_action_csv(dates, actions, amounts, save_path):
    """
    날짜별 행동 및 얼마나 csv 저장
    """
    df = pd.DataFrame({
        'date': dates,
        'action': actions,  # 1=충전, 0=대기, -1=방전
        'amount_kWh': amounts
    })
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

# def plot_action_line(dates, actions, amounts, save_path):
#     """
#     날짜별 충전/방전/대기 얼마나를 색상/마커로 구분하여 직관적으로 시각화
#     """
#     dates = np.array(dates)
#     actions = np.array(actions)
#     amounts = np.array(amounts)

#     plt.figure(figsize=(12, 5))

#     # 충전
#     mask_charge = actions == 1
#     plt.plot(dates[mask_charge], amounts[mask_charge], 'b^-', label='충전', markersize=8)

#     # 방전
#     mask_discharge = actions == -1
#     plt.plot(dates[mask_discharge], amounts[mask_discharge], 'rv-', label='방전', markersize=8)

#     # 대기
#     mask_wait = actions == 0
#     plt.plot(dates[mask_wait], amounts[mask_wait], 'ko-', label='대기', markersize=8)

#     plt.title('ESS 일별 에너지 운용(충전/방전/대기) 프로파일')
#     plt.xlabel('날짜')
#     plt.ylabel('kWh')
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

def print_cost_saving(total_kwh, price_per_kwh=280):
    """
    총 방전량 기반 비용 절감액 출력 (단가: MNT/kWh)
    """
    saving = total_kwh * price_per_kwh
    print(f"총 절감된 비용: {saving:,.0f} MNT")

