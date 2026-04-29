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
# 1. Path setup
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CBM_DIR = os.path.join(BASE_DIR, 'cbm')

WEATHER_PATH = os.path.join(DATA_DIR, '기상센서.csv')
POWER_PATH = os.path.join(DATA_DIR, 'wi2_0507_0605.csv')

os.makedirs(CBM_DIR, exist_ok=True)

def load_and_resample_1h(weather_path, power_path):
    print("Loading Data...")
    
    # --- 1. Load weather data ---
    df = pd.read_csv(weather_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])

    # --- 2. Load power data ---
    power_df = pd.read_csv(power_path, header=0, low_memory=False)
    timestamp_col = power_df.iloc[:, -1].astype(str).str.strip()
    power_df['power_TIMESTAMP'] = pd.to_datetime(timestamp_col, format='%Y-%m-%d_%H:%M:%S', errors='coerce')
    power_df = power_df.dropna(subset=['power_TIMESTAMP'])

    # Handle power column (index 43 or column name '1443')
    try:
        col_idx = 43
        power_df['Output Power'] = pd.to_numeric(power_df.iloc[:, col_idx], errors='coerce')
    except:
        if '1443' in power_df.columns:
            power_df['Output Power'] = pd.to_numeric(power_df['1443'], errors='coerce')
        else:
            raise ValueError("Power column (1443) not found.")

    # --- 3. Merge ---
    merged_df = pd.merge(df, power_df[['power_TIMESTAMP', 'Output Power']], 
                         left_on='TIMESTAMP', right_on='power_TIMESTAMP', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("No merged data found due to date-range mismatch.")

    # Numeric conversion
    cols = ['WS_Avg', 'WD_Avg', 'Temp_Avg', 'Air_P_Avg', 'Output Power']
    for col in cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # --- 4. 1-hour resampling ---
    merged_df = merged_df.set_index('TIMESTAMP')
    agg_rules = {
        'WS_Avg': 'mean', 'WD_Avg': 'mean', 'Temp_Avg': 'mean', 'Air_P_Avg': 'mean',
        'Output Power': 'max'
    }
    df_1h = merged_df.resample('1h').agg(agg_rules).dropna()

    # --- 5. Diff calculation with daily reset handling ---
    # Create date column
    df_1h['DATE_tmp'] = df_1h.index.date
    
    # Compute diff by day (prevents negative values from midnight reset)
    df_1h['Output Power Diff'] = df_1h.groupby('DATE_tmp')['Output Power'].diff()
    
    # For the first row after reset, fill NaN with the cumulative value at reset time
    df_1h['Output Power Diff'] = df_1h['Output Power Diff'].fillna(df_1h['Output Power'])

    # Count rows per day where power diff is <= 0
    zero_counts_per_day = df_1h[df_1h['Output Power Diff'] <= 0].groupby('DATE_tmp').size()
    
    # Extract dates with 20 or more zero/non-positive entries
    drop_dates = zero_counts_per_day[zero_counts_per_day >= 20].index
    
    # Remove those dates completely from the dataframe
    df_1h = df_1h[~df_1h['DATE_tmp'].isin(drop_dates)]

    # --- 6. Filtering (remove remaining rows where power is 0) ---
    df_1h = df_1h[df_1h['Output Power Diff'] > 0] 

    return df_1h.drop(columns=['DATE_tmp']).reset_index()

def create_features(df):
    """Create features (kept as-is)."""
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
    # Create timestamp for output files
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_model_file = os.path.join(CBM_DIR, f'wind_model_1h_{current_time}.cbm')
    current_feature_file = os.path.join(CBM_DIR, f'wind_features_1h_{current_time}.joblib')

    print(f"[{current_time}] Training process started...")
    
    try:
        # 1. Data preprocessing (apply reset logic and remove zeros)
        df = load_and_resample_1h(WEATHER_PATH, POWER_PATH)
        df = create_features(df)
        print(f"  -> Valid training rows: {len(df)}")

        if len(df) < 10:
             raise ValueError("Not enough data to train.")

        # 2. Set features and target
        target = 'Output Power Diff'
        features = [
            'WS_Avg', 'WD_sin', 'WD_cos', 'Temp_Avg', 'Air_P_Avg', 
            'WS_cubed', 'air_density', 'hour', 'month', 'WS_lag1', 'Power_lag1'
        ]

        X = df[features]
        y = df[target]

        # Split in chronological order to preserve time-series characteristics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 3. Define and train CatBoost model
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

        # 4. Compute and print evaluation metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n" + "="*50)
        print(" [Training completed - Model performance report]")

        print("-"*50)
        print(f" * R² Score (coefficient of determination): {r2:.4f}")

        print("="*50)

        # 5. Save model and feature list
        model.save_model(current_model_file)
        joblib.dump(features, current_feature_file)
        print(f"\n[Success] Model saved: {current_model_file}")

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
        print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()