import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import os

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CBM_DIR = os.path.join(BASE_DIR, 'cbm')
RESULT_DIR = os.path.join(BASE_DIR, 'wind_predict_1h_results')

WEATHER_PATH = os.path.join(DATA_DIR, '기상센서.csv')
POWER_PATH = os.path.join(DATA_DIR, 'wi2_0507_0605.csv')

MODEL_FILE = os.path.join(CBM_DIR, 'wind_model_1h.cbm')
FEATURE_FILE = os.path.join(CBM_DIR, 'wind_features_1h.joblib')

os.makedirs(RESULT_DIR, exist_ok=True)

def load_and_resample_inference(weather_path, power_path):
    # 기상 데이터 로드
    df = pd.read_csv(weather_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    
    # 실제값 병합 (검증용)
    if os.path.exists(power_path):
        power_df = pd.read_csv(power_path, header=0, low_memory=False)
        
        # 날짜 파싱 (언더바 처리 및 포맷 통일)
        timestamp_col = power_df.iloc[:, -1].astype(str).str.strip()
        power_df['power_TIMESTAMP'] = pd.to_datetime(timestamp_col, format='%Y-%m-%d_%H:%M:%S', errors='coerce')
        
        # 1443 컬럼 (인덱스 43)
        col_idx = 43
        power_df['Output Power'] = pd.to_numeric(power_df.iloc[:, col_idx], errors='coerce')
        
        df = pd.merge(df, power_df[['power_TIMESTAMP', 'Output Power']], 
                      left_on='TIMESTAMP', right_on='power_TIMESTAMP', how='left')

    # 숫자형 변환
    cols = ['WS_Avg', 'WD_Avg', 'Temp_Avg', 'Air_P_Avg', 'Output Power']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1시간 Resampling
    df = df.set_index('TIMESTAMP')
    agg_rules = {
        'WS_Avg': 'mean', 'WD_Avg': 'mean', 'Temp_Avg': 'mean', 'Air_P_Avg': 'mean'
    }
    
    # 누적 발전량은 구간 내 최대값 사용
    if 'Output Power' in df.columns:
        agg_rules['Output Power'] = 'max'
        
    df_1h = df.resample('1h').agg(agg_rules).dropna(subset=['WS_Avg'])
    
    # [핵심 수정] 리셋(음수 발생) 방지 로직 적용
    if 'Output Power' in df_1h.columns:
        # 차이 계산 (현재 누적값 - 이전 누적값)
        diff_val = df_1h['Output Power'].diff()
        
        # 0보다 작으면(리셋 발생), 차이값 대신 현재 누적값을 그대로 사용
        df_1h['ACTUAL_POWER'] = np.where(diff_val < 0, df_1h['Output Power'], diff_val)
    
    return df_1h.reset_index()

def create_features_inference(df):
    """특성 생성"""
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['month'] = df['TIMESTAMP'].dt.month
    df['WD_sin'] = np.sin(df['WD_Avg'] * np.pi / 180)
    df['WD_cos'] = np.cos(df['WD_Avg'] * np.pi / 180)
    df['WS_cubed'] = df['WS_Avg'] ** 3
    df['air_density'] = df['Air_P_Avg'] / (df['Temp_Avg'] + 273.15)
    
    # Lag Features
    df['WS_lag1'] = df['WS_Avg'].shift(1)
    
    if 'ACTUAL_POWER' in df.columns:
        df['Power_lag1'] = df['ACTUAL_POWER'].shift(1)
    else:
        df['Power_lag1'] = 0 

    return df

def main():
    if not os.path.exists(MODEL_FILE):
        return

    # 1. 데이터 로드
    df = load_and_resample_inference(WEATHER_PATH, POWER_PATH)
    df = create_features_inference(df)
    
    # 2. 모델 로드
    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_FILE)
    
    # 3. 예측
    valid_indices = df.dropna(subset=feature_cols).index
    X = df.loc[valid_indices, feature_cols]
    
    if len(X) > 0:
        preds = model.predict(X)
        preds = np.maximum(preds, 0) # 예측값 음수 방지
        
        # 4. 저장
        result_df = df.loc[valid_indices].copy()
        result_df['PREDICTED_POWER'] = preds
        
        save_cols = ['TIMESTAMP', 'PREDICTED_POWER']
        if 'ACTUAL_POWER' in result_df.columns:
            save_cols.append('ACTUAL_POWER')
            
        final_df = result_df[save_cols]
        
        csv_path = os.path.join(RESULT_DIR, 'wind_prediction_1h_result.csv')
        final_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()