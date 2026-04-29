import pandas as pd
import joblib
import os
import sys
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# ==========================================
# 1. 설정 (상대 경로 적용)
# ==========================================

# 현재 파일이 있는 폴더 경로 (예: D:/workspace/gpu/powermeter)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 추론용 데이터 경로
INFERENCE_DATA_PATHS = [
    os.path.join(BASE_DIR, 'data', 'powermeter_250520_250604.csv')
]

# 2. 모델 파일 경로
MODEL_PATH = os.path.join(BASE_DIR, 'pkl', 'lgbm_3days_1230_1137.pkl')

# 3. 결과 저장 경로
RESULT_SAVE_PATH = os.path.join(BASE_DIR, 'power_result', 'result.csv')

print(f"현재 실행 위치: {BASE_DIR}")
print(f"모델 읽을 경로: {MODEL_PATH}")

# ==========================================
# 2. 함수 정의
# ==========================================

def load_model(path):
    if not os.path.exists(path):
        print(f"🚨 오류: 모델 파일을 찾을 수 없습니다 -> {path}")
        raise FileNotFoundError(path)
    return joblib.load(path)

def preprocess_for_inference(paths, features):
    df_list = []
    for path in paths:
        if os.path.exists(path):
            temp = pd.read_csv(path)
            temp.columns = [c.strip() for c in temp.columns]
            df_list.append(temp)
        else:
            print(f"⚠️ 경고: 데이터 파일 없음 -> {path}")
            
    if not df_list:
        return pd.DataFrame()

    raw_df = pd.concat(df_list, axis=0, ignore_index=True)
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    raw_df = raw_df.sort_values('time').reset_index(drop=True)
    raw_df = raw_df.set_index('time')
    
    # 일별 데이터 집계
    daily_df = raw_df.resample('D').agg({
        'Ep-': 'max', 'P': 'mean', 'Ua': 'mean', 'Ia': 'mean'
    }).dropna()
    
    daily_df['daily_consumption'] = daily_df['Ep-'].diff()
    
    # Feature Engineering (학습 때와 동일해야 함)
    daily_df['lag_1d'] = daily_df['daily_consumption'].shift(1)
    daily_df['lag_2d'] = daily_df['daily_consumption'].shift(2)
    daily_df['lag_3d'] = daily_df['daily_consumption'].shift(3)
    daily_df['lag_7d'] = daily_df['daily_consumption'].shift(7)
    
    daily_df['roll_mean_3d'] = daily_df['daily_consumption'].shift(1).rolling(3).mean()
    daily_df['roll_mean_7d'] = daily_df['daily_consumption'].shift(1).rolling(7).mean()
    
    daily_df['dow'] = daily_df.index.dayofweek
    daily_df['month'] = daily_df.index.month
    
    inference_df = daily_df.dropna(subset=['lag_7d']).copy()
    return inference_df

if __name__ == "__main__":
    # 1. 모델 로드
    try:
        packet = load_model(MODEL_PATH)
    except FileNotFoundError:
        sys.exit()

    model = packet['model']
    features = packet['features']
    prediction_days = packet.get('prediction_days', 3)
    
    # LightGBM 버전 호환성 패치
    try:
        if hasattr(model, 'estimators_'):
            for est in model.estimators_:
                if hasattr(est, '_n_classes') and (est._n_classes is None):
                    est._n_classes = 1
                elif not hasattr(est, '_n_classes'):
                    est._n_classes = 1
    except:
        pass

    # 2. 데이터 처리
    try:
        df = preprocess_for_inference(INFERENCE_DATA_PATHS, features)
    except Exception as e:
        print(f"데이터 전처리 중 오류 발생: {e}")
        sys.exit()
    
    if len(df) == 0:
        print("전처리된 데이터가 없습니다. 종료합니다.")
        sys.exit()

    # 3. 예측 수행
    # 학습에 사용된 피처만 선택 (순서 중요)
    X = df[features]
    preds = model.predict(X)
    
    # 4. 결과 저장
    result_df = pd.DataFrame(index=df.index)
    
    for i in range(prediction_days):
        result_df[f'Pred_D+{i+1}'] = preds[:, i]
        
    result_df['Actual_Next_Day'] = df['daily_consumption'].shift(-1)
    
    result_df.index.name = 'Base_Date'
    result_df.reset_index(inplace=True)
    
    numeric_cols = result_df.select_dtypes(include=['float']).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(2)
    
    # 저장 폴더가 없으면 생성
    save_dir = os.path.dirname(RESULT_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    result_df.to_csv(RESULT_SAVE_PATH, index=False, encoding='utf-8-sig')
    # print(f"결과 저장 완료: {RESULT_SAVE_PATH}")