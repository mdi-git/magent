# 'sr_sum',              # current-time sr_sum
# 'sr_sum_d1',           # sr_sum at same hour on previous day
# 'power_60_sum_d1',     # power_60_sum at same hour on previous day
# 'hour',                # hour
# 'dayofweek',           # day of week
# 'sr_sum_rolling3_d1',  # previous-day rolling3
# 'sr_sum_rolling6_d1',  # previous-day rolling6
# 'sr_sum_cumsum_d1',    # previous-day cumulative sum
# 'sr_sum_diff1_d1'      # previous-day first-order diff

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

# ===== Training period and exclusion dates =====
train_periods = [
    ('2025-02-06', '2025-02-16'),
    # ('2025-03-01', '2025-03-15'),
]
exclude_dates = [
    '2025-02-10',
]

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'pv_1015_0605_hourly_sum.csv')
TARGET_PATH = os.path.join(BASE_DIR, 'data', 'sg0_60_0115_0605_hourly.csv')
WEIGHT_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(WEIGHT_DIR, exist_ok=True)

# 1. Load data
input_df = pd.read_csv(INPUT_PATH)
target_df = pd.read_csv(TARGET_PATH)
input_df['time'] = pd.to_datetime(input_df['time'])
target_df['datetime'] = pd.to_datetime(target_df['datetime'])

# After loading data
input_df['date_only'] = input_df['time'].dt.date
target_df['date_only'] = target_df['datetime'].dt.date

# Filter by training period
period_mask = False
for start, end in train_periods:
    period_mask |= ((input_df['time'] >= start) & (input_df['time'] <= end))
input_df = input_df[period_mask]

period_mask = False
for start, end in train_periods:
    period_mask |= ((target_df['datetime'] >= start) & (target_df['datetime'] <= end))
target_df = target_df[period_mask]

# Filter excluded dates
dates_to_exclude = [pd.to_datetime(d).date() for d in exclude_dates]
input_df = input_df[~input_df['date_only'].isin(dates_to_exclude)]
target_df = target_df[~target_df['date_only'].isin(dates_to_exclude)]

# Clean up date_only columns
input_df = input_df.drop(columns=['date_only'])
target_df = target_df.drop(columns=['date_only'])

# 2. Create time feature columns
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

# 8. Select features/target (temperature excluded)
feature_cols = [
    'sr_sum', 'sr_sum_d1', 'power_60_sum_d1', 'hour', 'dayofweek'
    # rolling/cumsum/diff1 are added below
]
target_col = 'power_60_sum'

# Compute daily rolling/cumsum/diff1
grouped = input_df.groupby(input_df['time'].dt.date)
input_df['sr_sum_rolling3'] = grouped['sr_sum'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
input_df['sr_sum_rolling6'] = grouped['sr_sum'].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
input_df['sr_sum_cumsum'] = grouped['sr_sum'].cumsum().reset_index(level=0, drop=True)
input_df['sr_sum_diff1'] = grouped['sr_sum'].diff(1).reset_index(level=0, drop=True)

# Shift by one day per hour to build previous-day features
for col in ['sr_sum_rolling3', 'sr_sum_rolling6', 'sr_sum_cumsum', 'sr_sum_diff1']:
    input_df[f'{col}_d1'] = input_df.groupby('hour')[col].shift(1)

# Add only previous-day columns to feature_cols
feature_cols += ['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']

input_df['sr_sum_ratio_d1'] = input_df['sr_sum'] / (input_df['sr_sum_d1'] + 1e-6)

input_df['target_time'] = input_df['time'] + pd.Timedelta(hours=1)

input_df = input_df[(input_df['hour'] >= 9) & (input_df['hour'] <= 18)]
target_df = target_df[(target_df['hour'] >= 9) & (target_df['hour'] <= 18)]

data = pd.merge(input_df, target_df[['datetime', 'power_60_sum', 'power_60_sum_d1']], left_on='target_time', right_on='datetime', how='inner')

# Replace missing values with 0 (including added features)
data[['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']] = data[['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']].fillna(0)

data = data.dropna(subset=feature_cols + [target_col])
X = data[feature_cols].values
y = data[target_col].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
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

# Stacking ensemble
stack = StackingRegressor(
    estimators=[
        ('cat', cat),
        ('lgbm', lgbm),
        ('xgb', xgb)
    ],
    final_estimator=Ridge(),
    n_jobs=1  # Use single process to avoid Windows non-ASCII path issues
)

# Train
stack.fit(X_scaled, y)

# Train individual models as well (for weight saving)
cat.fit(X_scaled, y)
lgbm.fit(X_scaled, y)
xgb.fit(X_scaled, y)

# Save
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

# Save feature list
joblib.dump(feature_cols, features_path)

print(f'Models and scaler saved: {WEIGHT_DIR}')

# Evaluate
preds = stack.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, preds))
print(f'Stacking ensemble RMSE on training data: {rmse:.4f}') 

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
