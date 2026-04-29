import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

LEARNING_RATE = float(os.getenv("MAGENT_LEARNING_RATE", "0.00001"))
N_ESTIMATORS = int(os.getenv("MAGENT_N_ESTIMATORS", "10000"))
NUM_LEAVES = int(os.getenv("MAGENT_NUM_LEAVES", "31"))
MAX_DEPTH = int(os.getenv("MAGENT_MAX_DEPTH", "-1"))
TRAIN_RESULT_PATH = os.getenv("MAGENT_TRAIN_RESULT_PATH")

# ==========================================
# 1. Setup
# ==========================================
DATA_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'powermeter_250520_250604.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'powermeter_250605_250609.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'powermeter_250611_250714.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'powermeter_250724_250913.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'powermeter_250913_251013.csv'),
]

# Model output directory and settings
PKL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pkl')
PREDICTION_DAYS = 3  # 3-day forecast

def load_and_preprocess(paths):
    df_list = []
    for path in paths:
        if os.path.exists(path):
            try:
                temp = pd.read_csv(path)
                temp.columns = [c.strip() for c in temp.columns]
                df_list.append(temp)
            except Exception:
                pass  # Skip quietly on load failure
    
    if not df_list:
        raise ValueError("Data loading failed: please check paths.")

    raw_df = pd.concat(df_list, axis=0, ignore_index=True)
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    raw_df = raw_df.sort_values('time').reset_index(drop=True)
    return raw_df

def create_dataset(raw_df):
    raw_df = raw_df.set_index('time')
    
    # Daily aggregation
    daily_df = raw_df.resample('D').agg({
        'Ep-': 'max', 'P': 'mean', 'Ua': 'mean', 'Ia': 'mean'
    }).dropna()
    
    # Consumption (diff)
    daily_df['daily_consumption'] = daily_df['Ep-'].diff()
    daily_df = daily_df[daily_df['daily_consumption'] > 0].copy()
    
    # Feature Engineering
    daily_df['lag_1d'] = daily_df['daily_consumption'].shift(1)
    daily_df['lag_2d'] = daily_df['daily_consumption'].shift(2)
    daily_df['lag_3d'] = daily_df['daily_consumption'].shift(3)
    daily_df['lag_7d'] = daily_df['daily_consumption'].shift(7)
    
    daily_df['roll_mean_3d'] = daily_df['daily_consumption'].shift(1).rolling(3).mean()
    daily_df['roll_mean_7d'] = daily_df['daily_consumption'].shift(1).rolling(7).mean()
    
    daily_df['dow'] = daily_df.index.dayofweek
    daily_df['month'] = daily_df.index.month

    # Target (Multi-output)
    target_cols = []
    for i in range(1, PREDICTION_DAYS + 1):
        col_name = f'target_day_{i}'
        daily_df[col_name] = daily_df['daily_consumption'].shift(-i)
        target_cols.append(col_name)
    
    model_df = daily_df.dropna()
    return model_df, target_cols

if __name__ == "__main__":
    # 1. Prepare data
    raw_df = load_and_preprocess(DATA_PATHS)
    train_df, target_cols = create_dataset(raw_df)
    
    print(f"Data Loaded. Samples: {len(train_df)}")

    features = ['daily_consumption', 'P', 'Ua', 'Ia', 
                'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d', 
                'roll_mean_3d', 'roll_mean_7d', 'dow', 'month']
    
    X = train_df[features]
    y = train_df[target_cols]
    
    # Train/validation split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 2. Train model
    print("Training LightGBM...")
    model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            max_depth=MAX_DEPTH,
            random_state=42,
            n_jobs=-1,
            verbose=100
        )
    )
    
    model.fit(X_train, y_train)
    
    # ------------------------------------------
    # Print validation metric (MAE)
    # ------------------------------------------
    preds = model.predict(X_test)
    
    print("\n" + "="*50)
    print("[ Validation Results (MAE) ]")
    print("="*50)
    
    # Overall MAE
    overall_mae = mean_absolute_error(y_test, preds)
    print(f"Total Average MAE : {overall_mae:.2f}")

    print("="*50 + "\n")
    # ------------------------------------------

    # ==========================================
    # 3. Save model (.pkl)
    # ==========================================
    os.makedirs(PKL_DIR, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lgbm_powermeter_3days_{current_time}.pkl"
    model_filepath = os.path.join(PKL_DIR, model_filename)

    packet = {
        "model": model,
        "features": features,
        "prediction_days": PREDICTION_DAYS,
        "learning_rate": LEARNING_RATE,
        "params": {
            "learning_rate": LEARNING_RATE,
            "n_estimators": N_ESTIMATORS,
            "num_leaves": NUM_LEAVES,
            "max_depth": MAX_DEPTH,
        },
        "mae": float(overall_mae),
    }
    joblib.dump(packet, model_filepath)
    print(f"Model saved: {model_filepath}")

    if TRAIN_RESULT_PATH:
        payload = {
            "learning_rate": LEARNING_RATE,
            "params": {
                "learning_rate": LEARNING_RATE,
                "n_estimators": N_ESTIMATORS,
                "num_leaves": NUM_LEAVES,
                "max_depth": MAX_DEPTH,
            },
            "metric_name": "mae",
            "score": float(overall_mae),
            "direction": "min",
            "model_path": model_filepath,
            "prediction_days": PREDICTION_DAYS,
        }
        with open(TRAIN_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)