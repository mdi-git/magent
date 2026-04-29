import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import itertools

# -----------------------------------------------------------
# 1. 설정 및 상수 정의
# -----------------------------------------------------------

# matplotlib 설정
#matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
#matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Malgun Gothic']

# 시스템 모드 설정
TEST_MODE = True
TEST_DATES = [
    "2025-05-08",
    "2025-05-09",
]

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'data', '기상센서.csv')
POWER_DATA_PATH = os.path.join(current_dir, 'data', 'wi2_0507_0605.csv')
MODEL_PATH = os.path.join(current_dir, 'cbm', 'wind_model_1_0822_1449.cbm')
OUTPUT_DIR = os.path.join(current_dir, 'wind_anomaly_results')
FEATURE_PATH = os.path.join(current_dir, 'cbm', 'feature_columns_0822_1449.joblib')

# -----------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------

def load_and_preprocess_data(file_path):
    """데이터 로드 및 기본 전처리"""
    
    # 1. 기상센서 데이터 로드
    df = pd.read_csv(file_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    
    # 2. 발전량 데이터 로드
    power_df = pd.read_csv(POWER_DATA_PATH, header=0, low_memory=False)
    
    # 3. 발전량 시간 포맷 통일
    power_df['TIMESTAMP_raw'] = power_df.iloc[:, -1].astype(str)
    power_df['TIMESTAMP_clean'] = power_df['TIMESTAMP_raw'].str.replace('_', ' ')
    power_df['power_TIMESTAMP'] = pd.to_datetime(power_df['TIMESTAMP_clean'], errors='coerce')
    
    # [핵심] 초(Seconds) 단위를 제거하여 기상센서(분 단위)와 매칭
    power_df['power_TIMESTAMP'] = power_df['power_TIMESTAMP'].dt.floor('min')
    
    # 4. 1443 컬럼(44번째, 발전량) 처리
    if '1443' in power_df.columns:
        power_col_name = '1443'
    else:
        power_col_name = power_df.columns[43]
    
    # 해당 컬럼을 숫자로 변환
    power_df['Output Power'] = pd.to_numeric(power_df[power_col_name], errors='coerce')
    power_df['Output Power Diff'] = power_df['Output Power'].diff()
    power_df = power_df.dropna(subset=['Output Power Diff', 'power_TIMESTAMP'])
    
    if len(power_df) == 0:
        return df
    
    # 5. 데이터 병합
    if 'Output Power Diff' in df.columns:
        df = df.drop(columns=['Output Power Diff'])
    
    df = pd.merge(df, power_df[['power_TIMESTAMP', 'Output Power', 'Output Power Diff']], 
                  left_on='TIMESTAMP', right_on='power_TIMESTAMP', how='inner')
    
    if len(df) == 0:
        return df
    
    # 6. 추가 특성 생성
    numeric_columns = ['Output Power Diff', 'WS_Avg', 'WD_Avg', 'Temp_Avg', 'Air_P_Avg']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['month'] = df['TIMESTAMP'].dt.month
    df['day'] = df['TIMESTAMP'].dt.day
    df['dayofweek'] = df['TIMESTAMP'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24],
                              labels=['night', 'morning', 'afternoon', 'evening'])
    df['time_of_day'] = df['time_of_day'].cat.codes
    
    return df

# -----------------------------------------------------------
# 3. 특성 공학 및 이상 감지 로직
# -----------------------------------------------------------

def create_advanced_features(df, seq_length):
    """고급 특성 생성"""
    if len(df) == 0:
        return df
        
    feature_list = ['WS_Avg', 'WD_Avg', 'Temp_Avg', 'Air_P_Avg', 'Output Power Diff']
    
    # 1. lag features
    for feature in feature_list:
        for i in range(1, seq_length + 1):
            df[f'{feature}_lag_{i}'] = df[feature].shift(i)
    
    # 2. Rolling features (24h)
    df['WS_Avg_rolling_mean_24h'] = df['WS_Avg'].rolling(window=24, min_periods=1).mean()
    df['WS_Avg_rolling_std_24h'] = df['WS_Avg'].rolling(window=24, min_periods=1).std()
    df['Output Power Diff_rolling_mean_24h'] = df['Output Power Diff'].rolling(window=24, min_periods=1).mean()
    
    # Previous day features
    df['WS_Avg_prev_day'] = df['WS_Avg'].shift(24)
    df['WD_Avg_prev_day'] = df['WD_Avg'].shift(24)
    df['Temp_Avg_prev_day'] = df['Temp_Avg'].shift(24)
    df['Air_P_Avg_prev_day'] = df['Air_P_Avg'].shift(24)
    df['Output Power Diff_prev_day'] = df['Output Power Diff'].shift(24)
    
    # Diff with prev day
    df['WS_Avg_diff_prev_day'] = df['WS_Avg'] - df['WS_Avg_prev_day']
    df['Temp_Avg_diff_prev_day'] = df['Temp_Avg'] - df['Temp_Avg_prev_day']
    df['Output Power Diff_diff_prev_day'] = df['Output Power Diff'] - df['Output Power Diff_prev_day']
    
    # 3. Wind Direction cyclic
    df['WD_sin'] = np.sin(df['WD_Avg'] * np.pi / 180)
    df['WD_cos'] = np.cos(df['WD_Avg'] * np.pi / 180)
    
    # 4. WS Category
    df['WS_category'] = pd.cut(df['WS_Avg'], bins=[0, 3, 7, 15, 25, 100], labels=[0, 1, 2, 3, 4])
    df['WS_category'] = df['WS_category'].cat.codes
    
    # 5. Air Density Proxy
    df['air_density_proxy'] = df['Air_P_Avg'] / (df['Temp_Avg'] + 273.15)
    
    # 6. WS Cubed
    df['WS_cubed'] = df['WS_Avg'] ** 3
    
    # 7. Change rate
    df['WS_change'] = df['WS_Avg'].diff()
    df['WD_change'] = df['WD_Avg'].diff()
    
    # 8. Seasonality
    df['season'] = ((df['month'] % 12 + 3) // 3).astype(int)
    
    df = df.dropna().reset_index(drop=True)
    return df

def apply_90min_continuous_anomaly_detection(df):
    """120분 연속된 이상 구간만 마스킹"""
    min_continuous_points = 6  
    is_anomaly = df['IS_ANOMALY'].values
    final_anomaly = np.zeros(len(df), dtype=int)
    
    start_idx = None
    for i, is_anom in enumerate(is_anomaly):
        if is_anom and start_idx is None:
            start_idx = i
        elif not is_anom and start_idx is not None:
            if i - start_idx >= min_continuous_points:
                final_anomaly[start_idx:i] = 1
            start_idx = None
    
    if start_idx is not None and len(df) - start_idx >= min_continuous_points:
        final_anomaly[start_idx:] = 1
    
    df['IS_ANOMALY'] = final_anomaly
    return df

# -----------------------------------------------------------
# 4. 시각화 함수
# -----------------------------------------------------------

def create_30min_plots(daily_data, output_dir, date):
    """결과 그래프 생성"""
    x_pos = np.arange(len(daily_data))
    
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 8), facecolor='#181b1f')  
    ax1.set_facecolor('#1a1a1a')
    ax1.grid(True, alpha=0.25, color='#383838', linestyle='-', linewidth=0.6)
    ax1.set_axisbelow(True)
    
    actual_power = daily_data['ACTUAL_POWER'].values
    predicted_power = daily_data['PREDICTED_POWER'].values
    
    ax1.plot(x_pos, actual_power, label='Actual Power', color='#FF6347', linewidth=3.5, alpha=0.9)
    ax1.plot(x_pos, predicted_power, label='Predicted Power', color='#52C7F2', linewidth=3.5, alpha=0.9)
    ax1.set_ylim(0, 2)
        
    # Anomaly Visualization
    error_mask = (daily_data['IS_ANOMALY'] == 1)
    start = None
    for i, val in enumerate(error_mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= 1:
                ax1.axvspan(x_pos[start]-0.5, x_pos[i-1]+0.5, color='#FFB74D', alpha=0.2, label='Anomaly Detection' if start == 0 else None, zorder=1)
            start = None
    if start is not None and len(daily_data) - start >= 1:
        ax1.axvspan(x_pos[start]-0.5, x_pos[len(daily_data)-1]+0.5, color='#FFB74D', alpha=0.2, label='Anomaly Detection' if start == 0 else None, zorder=1)
    
    # Legend Handling
    import matplotlib.patches as mpatches
    dummy_actual = mpatches.Patch(color='#FF6347', alpha=0.9, label='Actual Power')
    dummy_pred = mpatches.Patch(color='#52C7F2', alpha=0.9, label='Predicted Power')
    dummy_anom = mpatches.Patch(color='#FFB74D', alpha=0.2, label='Anomaly Detection')
    
    handles, labels = ax1.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    
    temp_dict = {}
    for h, l in zip(handles, labels):
        if l not in temp_dict:
            temp_dict[l] = h
            new_handles.append(h)
            new_labels.append(l)
            
    for dummy, name in zip([dummy_actual, dummy_pred, dummy_anom], ['Actual Power', 'Predicted Power', 'Anomaly Detection']):
        if name not in new_labels:
            new_handles.append(dummy)
            new_labels.append(name)
    
    ax1.legend(new_handles, new_labels, loc='upper left', fontsize=14, framealpha=0.98, 
               facecolor='#262626', edgecolor='#555555', labelcolor='#f0f0f0')
    
    plt.tight_layout()
    ax1.set_title(f'{date}', fontsize=20, fontweight='600', color='#f5f5f5')
    ax1.set_ylabel('Power Generation (kWh)', fontsize=14, fontweight='normal', color='#aaaaaa')
    
    ax1.set_xticks(x_pos)
    hour_labels = []
    for i, timestamp in enumerate(daily_data['TIMESTAMP']):
        if timestamp.minute % 60 == 0:
            hour_labels.append(timestamp.strftime('%H:%M'))
        else:
            hour_labels.append('')
    
    ax1.set_xticklabels(hour_labels, rotation=45, fontsize=13, fontweight='bold', color='#ffffff')
    ax1.tick_params(axis='y', labelsize=13, labelcolor='#ffffff')
    ax1.tick_params(axis='x', labelsize=13, labelcolor='#ffffff')
    
    for spine in ax1.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    graph_path = os.path.join(output_dir, f'anomaly_report_{date}.png')
    plt.savefig(graph_path, dpi=200, bbox_inches='tight', facecolor='#181b1f', edgecolor='none')  
    plt.close()

# -----------------------------------------------------------
# 5. 메인 함수
# -----------------------------------------------------------

def main():
    """메인 실행 함수"""
    try:
        if TEST_MODE:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # 1. 데이터 로드 및 전처리
            df = load_and_preprocess_data(DATA_PATH)
            
            # [안전장치] 로드된 데이터의 TIMESTAMP 확인
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
            df = df.dropna(subset=['TIMESTAMP'])
            
            if len(df) == 0:
                return

            # 2. 모델 로드
            feature_columns = joblib.load(FEATURE_PATH)
            model = CatBoostRegressor()
            model.load_model(MODEL_PATH)
            
            # 3. 날짜별 시뮬레이션
            for date_str in TEST_DATES:
                TEST_NOW = date_str
                test_day = pd.to_datetime(TEST_NOW).normalize()
                
                start_time = test_day
                day_df = df[df['TIMESTAMP'].dt.date == test_day.date()].copy()
                
                if len(day_df) > 0:
                    end_time = day_df['TIMESTAMP'].max()
                else:
                    end_time = test_day + pd.Timedelta(hours=23, minutes=59)
                
                target_df = df[(df['TIMESTAMP'] >= start_time - pd.Timedelta(minutes=10)) & (df['TIMESTAMP'] <= end_time)].copy()
                
                if len(target_df) == 0:
                    continue
                
                # 피처 생성
                target_df = create_advanced_features(target_df, seq_length=6)
                target_df = target_df.dropna(subset=feature_columns)
                
                if len(target_df) > 0:
                    current_time = start_time
                    records = []
                    
                    while current_time <= end_time:
                        actual = np.nan
                        pred = 0.0
                        ws = 0.0
                        
                        target_time = current_time - pd.Timedelta(minutes=5)
                        target_row = target_df[target_df['TIMESTAMP'] <= target_time]
                        
                        if len(target_row) > 0:
                            last_row = target_row.iloc[[-1]]
                            original_pred = float(np.abs(model.predict(last_row[feature_columns]))[0]) * 0.7
                            
                            actual_row = target_df[target_df['TIMESTAMP'] == target_time]
                            if len(actual_row) > 0:
                                actual = float(actual_row.iloc[0]['Output Power Diff']) if 'Output Power Diff' in actual_row.columns else np.nan
                                ws = float(actual_row.iloc[0]['WS_Avg']) if 'WS_Avg' in actual_row.columns else 0.0
                                
                                if actual == 0 or pd.isna(actual):
                                    if ws > 7:
                                        wind_based_pred = (ws ** 3) * 0.001
                                        pred = max(original_pred, wind_based_pred)
                                    else:
                                        pred = original_pred
                                else:
                                    pred = original_pred
                            else:
                                pred = min(original_pred, 4.0)
                        
                            diff = abs(actual - pred)
                            
                            diff_anomaly = (diff > 15) and (ws > 8)
                            zero_power_anomaly = (actual == 0 or pd.isna(actual)) and (ws > 3)
                            large_diff_anomaly = (pred - actual) > 10.0 and (ws > 5)
                            
                            if actual > pred:
                                is_anom = 0
                            else:
                                is_anom = int(diff_anomaly or zero_power_anomaly or large_diff_anomaly)
                            
                            records.append({'TIMESTAMP': current_time, 'ACTUAL_POWER': actual/10, 'PREDICTED_POWER': pred/1, 'IS_ANOMALY': is_anom})
                        
                        current_time += pd.Timedelta(minutes=1)
                    
                    if records:
                        sim_df = pd.DataFrame(records)
                        sim_df = sim_df.set_index('TIMESTAMP')
                        agg = sim_df.resample('20min').agg({'ACTUAL_POWER':'mean','PREDICTED_POWER':'mean','IS_ANOMALY':'mean'}).reset_index()
                        agg['IS_ANOMALY'] = (agg['IS_ANOMALY'] >= 0.5).astype(int)
                        
                        agg = apply_90min_continuous_anomaly_detection(agg)
                        create_30min_plots(agg, OUTPUT_DIR, test_day.date())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()