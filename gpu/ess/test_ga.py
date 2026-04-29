import os
import sys
import torch
import pandas as pd
import numpy as np
from env import ESSEnv

# ==============================================================================
# 1. Setup
# ==============================================================================
TARGET_YEAR = 2025        # target year

START_DATE = f'{TARGET_YEAR}-05-01'
END_DATE = f'{TARGET_YEAR}-05-31'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# File path setup

DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'ess_carbon.csv')
WEIGHT_PATH = os.path.join(CURRENT_DIR, 'pt', 'ga_best_individual_0724_1401.pt')
RESULT_DIR = os.path.join(CURRENT_DIR, 'result')
CSV_PATH = os.path.join(RESULT_DIR, 'ga_test_action.csv')

# ==============================================================================
# 2. Function definitions
# ==============================================================================

def load_data(data_path, start_date, end_date, target_year=None):
    """Load data and filter by date range (with optional year override)."""
    if not os.path.exists(data_path):
        print(f"Error: data file does not exist -> {data_path}")
        sys.exit()
        
    df = pd.read_csv(data_path)
    
    # Year override logic
    if target_year is not None:
        # If CSV uses MM-DD format, prepend target year
        df['datetime'] = pd.to_datetime(str(target_year) + '-' + df['datetime'], format='%Y-%m-%d')

    else:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Date filtering

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
    return df.reset_index(drop=True)

def save_action_csv(dates, actions, amounts, save_path):
    """Save result CSV."""

    df = pd.DataFrame({
        'date': dates,
        'action': actions,  # 1=charge, 0=idle, -1=discharge

        'amount_kWh': amounts
    })
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

def print_carbon_saving(total_kwh):
    """Print carbon reduction."""

    carbon_factor = 0.4448
    saved = total_kwh * carbon_factor * 0.001
    print(f"-"*30)
    print(f"Total carbon reduction: {saved:.2f} tCO2-eq")

    print(f"-"*30)

# ==============================================================================
# 3. Main execution logic
# ==============================================================================

if __name__ == "__main__":
    # Create output directory

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if not os.path.exists(WEIGHT_PATH):
        print(f"Error: weight file does not exist -> {WEIGHT_PATH}")

        sys.exit()

    print(f">>> Simulation start: {START_DATE} ~ {END_DATE}")

    # 1. Load data

    df = load_data(DATA_PATH, START_DATE, END_DATE, target_year=TARGET_YEAR)
    N_DAYS = len(df)
    
    if N_DAYS == 0:
        print("!!! Error: no data for the selected period. Please check dates.")

        sys.exit()
        
    env = ESSEnv(df)

    # 2. Load model (weights)
    print(f">>> Loading model: {os.path.basename(WEIGHT_PATH)}")

    try:
        data = torch.load(WEIGHT_PATH, weights_only=False)
        actions = data['actions']
        amounts = data['amounts']
    except Exception as e:
        print(f"!!! Model load failed: {e}")

        sys.exit()

    # 3. Run loop

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
        
        # Date format (MM-dd)

        dates.append(df.iloc[i]['datetime'].strftime('%m-%d'))
        actions_list.append(action)
        amounts_list.append(amount)
        
        if action == 1:
            total_charge_kwh += amount
            
        state, _, done, _ = env.step(action, amount)
        if done:
            break

    # 4. Save results

    save_action_csv(dates, actions_list, amounts_list, CSV_PATH)
    

    # 5. Final output

    print_carbon_saving(total_charge_kwh)