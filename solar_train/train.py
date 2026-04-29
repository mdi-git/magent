# 'sr_sum',              # 현재 시점 sr_sum
# 'sr_sum_d1',           # 전날 같은 시각 sr_sum
# 'power_60_sum_d1',     # 전날 같은 시각 power_60_sum
# 'hour',                # 시간
# 'dayofweek',           # 요일
# 'sr_sum_rolling3_d1',  # 전날 rolling3
# 'sr_sum_rolling6_d1',  # 전날 rolling6
# 'sr_sum_cumsum_d1',    # 전날 누적합
# 'sr_sum_diff1_d1'      # 전날 1시계차

import os
import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime


LEARNING_RATE = float(os.getenv("MAGENT_LEARNING_RATE", "0.1"))
CAT_DEPTH = int(os.getenv("MAGENT_CAT_DEPTH", "6"))
LGBM_N_ESTIMATORS = int(os.getenv("MAGENT_LGBM_N_ESTIMATORS", "200"))
XGB_N_ESTIMATORS = int(os.getenv("MAGENT_XGB_N_ESTIMATORS", "300"))
XGB_MAX_DEPTH = int(os.getenv("MAGENT_XGB_MAX_DEPTH", "6"))
TRAIN_RESULT_PATH = os.getenv("MAGENT_TRAIN_RESULT_PATH")

# ===== 학습 기간 및 제외일 설정 =====
train_periods = [
    ('2025-02-06', '2025-02-16'),
    # ('2025-03-01', '2025-03-15'),
]
exclude_dates = [
    '2025-02-10',
]

# 데이터 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'pv_1015_0605_hourly_sum.csv')
TARGET_PATH = os.path.join(BASE_DIR, 'data', 'sg0_60_0115_0605_hourly.csv')
WEIGHT_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(WEIGHT_DIR, exist_ok=True)

# 1. 데이터 로드
input_df = pd.read_csv(INPUT_PATH)
target_df = pd.read_csv(TARGET_PATH)
input_df['time'] = pd.to_datetime(input_df['time'])
target_df['datetime'] = pd.to_datetime(target_df['datetime'])

# 데이터 로드 후
input_df['date_only'] = input_df['time'].dt.date
target_df['date_only'] = target_df['datetime'].dt.date

# 학습 구간 필터링
period_mask = False
for start, end in train_periods:
    period_mask |= ((input_df['time'] >= start) & (input_df['time'] <= end))
input_df = input_df[period_mask]

period_mask = False
for start, end in train_periods:
    period_mask |= ((target_df['datetime'] >= start) & (target_df['datetime'] <= end))
target_df = target_df[period_mask]

# 제외일 필터링
dates_to_exclude = [pd.to_datetime(d).date() for d in exclude_dates]
input_df = input_df[~input_df['date_only'].isin(dates_to_exclude)]
target_df = target_df[~target_df['date_only'].isin(dates_to_exclude)]

# date_only 컬럼 정리
input_df = input_df.drop(columns=['date_only'])
target_df = target_df.drop(columns=['date_only'])

# 2. 시간 정보 컬럼 생성
input_df['hour'] = input_df['time'].dt.hour
input_df['dayofweek'] = input_df['time'].dt.dayofweek
input_df['date'] = input_df['time'].dt.date
target_df['hour'] = target_df['datetime'].dt.hour
target_df['date'] = target_df['datetime'].dt.date

def get_prevday_value(df, col, time_col='time'):
    df = df.copy()
    df['prevday'] = df[time_col] - pd.Timedelta(days=1)
    prev = df[[time_col, col]].copy()
    prev[time_col] = prev[time_col] + pd.Timedelta(days=1)
    prev = prev.rename(columns={col: f'{col}_d1'})
    return pd.merge(df, prev[[time_col, f'{col}_d1']], left_on=time_col, right_on=time_col, how='left')

input_df = get_prevday_value(input_df, 'sr_sum', 'time')
target_df = get_prevday_value(target_df, 'power_60_sum', 'datetime')

# 8. feature/target 선택 (온도값 제외)
feature_cols = [
    'sr_sum', 'sr_sum_d1', 'power_60_sum_d1', 'hour', 'dayofweek'
    # rolling/cumsum/diff1 등은 아래에서 추가
]
target_col = 'power_60_sum'

# 날짜별 rolling/cumsum/diff1 계산
grouped = input_df.groupby(input_df['time'].dt.date)
input_df['sr_sum_rolling3'] = grouped['sr_sum'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
input_df['sr_sum_rolling6'] = grouped['sr_sum'].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
input_df['sr_sum_cumsum'] = grouped['sr_sum'].cumsum().reset_index(level=0, drop=True)
input_df['sr_sum_diff1'] = grouped['sr_sum'].diff(1).reset_index(level=0, drop=True)

# hour별로 하루 shift해서 전날 값으로 만듦
for col in ['sr_sum_rolling3', 'sr_sum_rolling6', 'sr_sum_cumsum', 'sr_sum_diff1']:
    input_df[f'{col}_d1'] = input_df.groupby('hour')[col].shift(1)

# feature_cols에 전날 컬럼만 추가
feature_cols += ['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']

input_df['sr_sum_ratio_d1'] = input_df['sr_sum'] / (input_df['sr_sum_d1'] + 1e-6)

input_df['target_time'] = input_df['time'] + pd.Timedelta(hours=1)

input_df = input_df[(input_df['hour'] >= 9) & (input_df['hour'] <= 18)]
target_df = target_df[(target_df['hour'] >= 9) & (target_df['hour'] <= 18)]

data = pd.merge(input_df, target_df[['datetime', 'power_60_sum', 'power_60_sum_d1']], left_on='target_time', right_on='datetime', how='inner')

# 결측치 0으로 대체 (추가된 feature 포함)
data[['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']] = data[['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']].fillna(0)

data = data.dropna(subset=feature_cols + [target_col])
X = data[feature_cols].values
y = data[target_col].values

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 모델 정의
cat = CatBoostRegressor(
    verbose=0,
    random_seed=1004,
    learning_rate=LEARNING_RATE,
    depth=CAT_DEPTH,
)
lgbm = LGBMRegressor(
    random_state=1004,
    learning_rate=LEARNING_RATE,
    n_estimators=LGBM_N_ESTIMATORS,
)
xgb = XGBRegressor(
    random_state=1004,
    verbosity=0,
    learning_rate=LEARNING_RATE,
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
)

# 스태킹 앙상블
stack = StackingRegressor(
    estimators=[
        ('cat', cat),
        ('lgbm', lgbm),
        ('xgb', xgb)
    ],
    final_estimator=Ridge(),
    n_jobs=1  # 병렬 처리 대신 단일 프로세스 사용 (윈도우 한글 경로 에러 방지)
)

# 학습
stack.fit(X_scaled, y)

# 개별 모델도 따로 학습 (가중치 저장용)
cat.fit(X_scaled, y)
lgbm.fit(X_scaled, y)
xgb.fit(X_scaled, y)

# 저장
now = datetime.now().strftime('%m%d_%H%M%S')
stack_path = os.path.join(WEIGHT_DIR, f'ensemble_stack_{now}.pkl')
cat_path = os.path.join(WEIGHT_DIR, f'ensemble_cat_{now}.pkl')
lgbm_path = os.path.join(WEIGHT_DIR, f'ensemble_lgbm_{now}.pkl')
xgb_path = os.path.join(WEIGHT_DIR, f'ensemble_xgb_{now}.pkl')
scaler_path = os.path.join(WEIGHT_DIR, f'ensemble_scaler_{now}.pkl')
features_path = os.path.join(WEIGHT_DIR, f'ensemble_features_{now}.pkl')

joblib.dump(stack, stack_path)
joblib.dump(cat, cat_path)
joblib.dump(lgbm, lgbm_path)
joblib.dump(xgb, xgb_path)
joblib.dump(scaler, scaler_path)

# feature 목록도 저장
joblib.dump(feature_cols, features_path)

print(f'모델 및 스케일러 저장 완료: {WEIGHT_DIR}')

# 평가
preds = stack.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, preds))
print(f'학습 데이터 스태킹 앙상블 RMSE: {rmse:.4f}') 

if TRAIN_RESULT_PATH:
    payload = {
        "learning_rate": LEARNING_RATE,
        "params": {
            "learning_rate": LEARNING_RATE,
            "cat_depth": CAT_DEPTH,
            "lgbm_n_estimators": LGBM_N_ESTIMATORS,
            "xgb_n_estimators": XGB_N_ESTIMATORS,
            "xgb_max_depth": XGB_MAX_DEPTH,
        },
        "metric_name": "rmse",
        "score": float(rmse),
        "direction": "min",
        "stack_path": stack_path,
        "scaler_path": scaler_path,
        "features_path": features_path,
    }
    with open(TRAIN_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
