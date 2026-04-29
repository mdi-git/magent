import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LEARNING_RATE = float(os.getenv("MAGENT_LEARNING_RATE", "0.04"))
ITERATIONS = int(os.getenv("MAGENT_ITERATIONS", "100000"))
DEPTH = int(os.getenv("MAGENT_DEPTH", "2"))
L2_LEAF_REG = float(os.getenv("MAGENT_L2_LEAF_REG", "3"))
TRAIN_RESULT_PATH = os.getenv("MAGENT_TRAIN_RESULT_PATH")

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CBM_DIR = os.path.join(BASE_DIR, 'cbm')

WEATHER_PATH = os.path.join(DATA_DIR, '기상센서.csv')
POWER_PATH = os.path.join(DATA_DIR, 'wi2_0507_0605.csv')

os.makedirs(CBM_DIR, exist_ok=True)

def load_and_resample_1h(weather_path, power_path):
    print("Loading Data...")
    
    # --- 1. 기상 데이터 로드 ---
    df = pd.read_csv(weather_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])

    # --- 2. 발전량 데이터 로드 ---
    power_df = pd.read_csv(power_path, header=0, low_memory=False)
    timestamp_col = power_df.iloc[:, -1].astype(str).str.strip()
    power_df['power_TIMESTAMP'] = pd.to_datetime(timestamp_col, format='%Y-%m-%d_%H:%M:%S', errors='coerce')
    power_df = power_df.dropna(subset=['power_TIMESTAMP'])

    # 발전량 컬럼 처리 (인덱스 43 또는 이름 '1443')
    try:
        col_idx = 43
        power_df['Output Power'] = pd.to_numeric(power_df.iloc[:, col_idx], errors='coerce')
    except:
        if '1443' in power_df.columns:
            power_df['Output Power'] = pd.to_numeric(power_df['1443'], errors='coerce')
        else:
            raise ValueError("발전량 컬럼(1443)을 찾을 수 없습니다.")

    # --- 3. 병합 ---
    merged_df = pd.merge(df, power_df[['power_TIMESTAMP', 'Output Power']], 
                         left_on='TIMESTAMP', right_on='power_TIMESTAMP', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("날짜 범위 불일치로 병합 데이터가 없습니다.")

    # 숫자형 변환
    cols = ['WS_Avg', 'WD_Avg', 'Temp_Avg', 'Air_P_Avg', 'Output Power']
    for col in cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # --- 4. 1시간 Resampling ---
    merged_df = merged_df.set_index('TIMESTAMP')
    agg_rules = {
        'WS_Avg': 'mean', 'WD_Avg': 'mean', 'Temp_Avg': 'mean', 'Air_P_Avg': 'mean',
        'Output Power': 'max'
    }
    df_1h = merged_df.resample('1h').agg(agg_rules).dropna()

    # --- 5. 하루 단위 리셋 대응 차분 계산 ---
    # 날짜 컬럼 생성
    df_1h['DATE_tmp'] = df_1h.index.date
    
    # 날짜별로 그룹을 묶어 차분 계산 (자정 리셋 시 음수 발생 방지)
    df_1h['Output Power Diff'] = df_1h.groupby('DATE_tmp')['Output Power'].diff()
    
    # 리셋 후 첫 번째 데이터(NaN)는 리셋된 시점의 누적값이 곧 해당 시간 발전량이므로 그대로 채움
    df_1h['Output Power Diff'] = df_1h['Output Power Diff'].fillna(df_1h['Output Power'])

    # 발전량(차분)이 0 이하인 데이터의 개수를 날짜별로 카운트
    zero_counts_per_day = df_1h[df_1h['Output Power Diff'] <= 0].groupby('DATE_tmp').size()
    
    # 0값이 20개 이상인 날짜 목록 추출
    drop_dates = zero_counts_per_day[zero_counts_per_day >= 20].index
    
    # 해당 날짜를 데이터프레임에서 완전히 제외
    df_1h = df_1h[~df_1h['DATE_tmp'].isin(drop_dates)]

    # --- 6. 필터링 (발전량이 0인 나머지 개별 데이터 제외) ---
    df_1h = df_1h[df_1h['Output Power Diff'] > 0] 

    return df_1h.drop(columns=['DATE_tmp']).reset_index()

def create_features(df):
    """특성 생성 (기존 유지)"""
    if len(df) == 0: return df
    
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['month'] = df['TIMESTAMP'].dt.month
    df['WD_sin'] = np.sin(df['WD_Avg'] * np.pi / 180)
    df['WD_cos'] = np.cos(df['WD_Avg'] * np.pi / 180)
    df['WS_cubed'] = df['WS_Avg'] ** 3
    df['air_density'] = df['Air_P_Avg'] / (df['Temp_Avg'] + 273.15)
    
    # Lag Features
    df['WS_lag1'] = df['WS_Avg'].shift(1)
    df['Power_lag1'] = df['Output Power Diff'].shift(1)
    
    return df.dropna().reset_index(drop=True)

def main():
    # 저장용 타임스탬프 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_model_file = os.path.join(CBM_DIR, f'wind_model_1h_{current_time}.cbm')
    current_feature_file = os.path.join(CBM_DIR, f'wind_features_1h_{current_time}.joblib')

    print(f"[{current_time}] 학습 프로세스 시작...")
    
    try:
        # 1. 데이터 전처리 (리셋 로직 및 0 제외 적용)
        df = load_and_resample_1h(WEATHER_PATH, POWER_PATH)
        df = create_features(df)
        print(f"  -> 유효 학습 데이터 수: {len(df)} 행")

        if len(df) < 10:
             raise ValueError("데이터가 너무 적어 학습을 진행할 수 없습니다.")

        # 2. 특징 및 타겟 설정
        target = 'Output Power Diff'
        features = [
            'WS_Avg', 'WD_sin', 'WD_cos', 'Temp_Avg', 'Air_P_Avg', 
            'WS_cubed', 'air_density', 'hour', 'month', 'WS_lag1', 'Power_lag1'
        ]

        X = df[features]
        y = df[target]

        # 시계열 특성을 고려하여 순서대로 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 3. CatBoost 모델 정의 및 학습
        print("Training CatBoost Regressor...")
        model = CatBoostRegressor(
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            depth=DEPTH,
            l2_leaf_reg=L2_LEAF_REG,
            loss_function='RMSE',
            verbose=2000,

        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=1000)

        # 4. 평가지표 산출 및 출력
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n" + "="*50)
        print(" [학습 완료 - 모델 성능 리포트]")
        print("-"*50)
        print(f" * R² Score (결정 계수): {r2:.4f}")
        print("="*50)

        # 5. 모델 및 특징 리스트 저장
        model.save_model(current_model_file)
        joblib.dump(features, current_feature_file)
        print(f"\n[성공] 모델 저장됨: {current_model_file}")

        if TRAIN_RESULT_PATH:
            payload = {
                "learning_rate": LEARNING_RATE,
                "params": {
                    "learning_rate": LEARNING_RATE,
                    "iterations": ITERATIONS,
                    "depth": DEPTH,
                    "l2_leaf_reg": L2_LEAF_REG,
                },
                "metric_name": "r2",
                "score": float(r2),
                "direction": "max",
                "rmse": float(rmse),
                "mae": float(mae),
                "model_path": current_model_file,
                "feature_path": current_feature_file,
            }
            with open(TRAIN_RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"\n[오류 발생] {e}")

if __name__ == "__main__":
    main()