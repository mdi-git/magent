import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os
import matplotlib.pyplot as plt
import matplotlib

# matplotlib settings
#matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

# Configuration
TEST_MODE = True
# Single test date
TEST_DATE = "2025-03-03 00:00:00"
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'data', 'pre_pv_1015_0707.csv')
TARGET_PATH = os.path.join(current_dir, 'data', 'sg0_2025-01-15_06-05.csv')
MODEL_PATH = os.path.join(current_dir, 'cbm', 'solar_model_sg_0714_1835.cbm')
OUTPUT_DIR = os.path.join(current_dir, 'solar_anomaly_results')


def load_data():
    """Load and preprocess data."""
    pv_data = pd.read_csv(DATA_PATH, low_memory=False).rename(columns={'time': 'TIMESTAMP', 'temp': 'surface_temp'})
    target_data = pd.read_csv(TARGET_PATH, header=0, low_memory=False)  # Use first row as header
    
   
    # Compute diff of column 63 and multiply by 0.1 to create power_63
    target_data['63'] = pd.to_numeric(target_data['63'], errors='coerce')
    target_data['power_63'] = target_data['63'].diff() * 0.1
    target_data = target_data.dropna()  # Remove first NaN row
    
    # Convert datetime column to TIMESTAMP (replace underscore with space)
    target_data['datetime_clean'] = target_data['datetime'].str.replace('_', ' ')
    target_data['TIMESTAMP'] = pd.to_datetime(target_data['datetime_clean'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Count rows with parse failures
    failed_parse_count = target_data['TIMESTAMP'].isna().sum()
    
    target_data = target_data.dropna(subset=['TIMESTAMP'])  # Drop rows with invalid timestamps
    
    for df in [pv_data]:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    target_data['hour'] = target_data['TIMESTAMP'].dt.hour
    
    return pv_data, target_data


def create_features(merged):
    """Create features."""

    merged['power_63'] = merged['power_63'] * 10
    # merged['power_63'] = merged['power_63'] 
    
    merged['sin_hour'] = np.sin(2 * np.pi * merged['hour'] / 24)
    merged['cos_hour'] = np.cos(2 * np.pi * merged['hour'] / 24)
    for i in range(1, 4):
        merged[f'power_lag{i}'] = merged['power_63'].shift(i)
    merged['power_per_sr'] = merged['power_63'] / (merged['sr'] + 1e-6)
    return merged.dropna().reset_index(drop=True)


def apply_60min_continuous_anomaly_detection(df):
    """Detect anomalies only when they persist for 60 minutes continuously."""
    is_anomaly = df['IS_ANOMALY'].values
    final_anomaly = np.zeros(len(df), dtype=int)
    start_idx = None
    
    for i, is_anom in enumerate(is_anomaly):
        if is_anom and start_idx is None:
            start_idx = i
        elif not is_anom and start_idx is not None:
            if i - start_idx >= 12:  # 5 min × 12 = 60 min
                final_anomaly[start_idx:i] = 1
            start_idx = None
    
    if start_idx is not None and len(df) - start_idx >= 12:
        final_anomaly[start_idx:] = 1
    
    df['IS_ANOMALY'] = final_anomaly
    return df


def create_daily_plots(result, output_dir):
    """Create daily plots."""
    result['date'] = pd.to_datetime(result['TIMESTAMP']).dt.date
    result['TIMESTAMP'] = pd.to_datetime(result['TIMESTAMP'])
    
    for date in sorted(result['date'].unique()):
        daily_data = result[result['date'] == date].copy().sort_values('TIMESTAMP')
        if len(daily_data) == 0:
            continue
            
            
        # Aggregate by 10-minute intervals
        daily_10min = daily_data.set_index('TIMESTAMP').resample('10min').agg({
            'ACTUAL_POWER': 'sum', 'PREDICTED_POWER': 'sum', 'ERROR_RATE': 'mean'
        }).dropna().reset_index()
        
        if len(daily_10min) == 0:
            continue
            
        # Determine scale
        max_power = max(daily_10min[['ACTUAL_POWER', 'PREDICTED_POWER']].max())
        use_log_scale = max_power < 11  # keep log scale disabled
        use_log_scale = False
        
        # Compute upper/lower bands (±20% around predicted generation)
        daily_10min['UPPER_BAND'] = daily_10min['PREDICTED_POWER'] * 1.1
        daily_10min['LOWER_BAND'] = daily_10min['PREDICTED_POWER'] * 0.9
        
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 6), facecolor='#181b1f')  # 8:3 ratio
        ax1.set_facecolor('#1a1a1a')  # deeper dark background
        
        ax1.grid(True, alpha=0.25, color='#383838', linestyle='-', linewidth=0.6)
        ax1.set_axisbelow(True)
        
        x_pos = np.arange(len(daily_10min))
        
        # Main plot
        ax1.plot(x_pos, daily_10min['PREDICTED_POWER'], label='Predicted Power', 
                color='#0066FF', linewidth=1.2, alpha=1.0, zorder=3, linestyle='-')  # vivid blue (predicted)
        ax1.plot(x_pos, daily_10min['ACTUAL_POWER'], label='Actual Power', 
                color='#FF0033', linewidth=1.2, alpha=1.0, zorder=2, linestyle='-')  # vivid red (actual)
        
        # Upper/lower band guide lines (fully transparent)
        ax1.plot(x_pos, daily_10min['UPPER_BAND'], color='yellow', linestyle='--', linewidth=3, alpha=0)
        ax1.plot(x_pos, daily_10min['LOWER_BAND'], color='yellow', linestyle='--', linewidth=3, alpha=0)
        
        # Anomaly segments
        anomaly_mask = create_anomaly_mask(daily_10min)
        for start, end in get_consecutive_ranges(anomaly_mask):
            ax1.axvspan(x_pos[start]-0.5, x_pos[end]+0.5, 
                       color='#FF8C42', alpha=0.4, zorder=1)  
        
        # Normal segments
        normal_mask = (daily_10min['ERROR_RATE'] <= 12) & (~anomaly_mask)
        for start, end in get_consecutive_ranges(normal_mask):

            pred_power = daily_10min['PREDICTED_POWER'].iloc[start:end+1]
            band_width = (daily_10min['UPPER_BAND'].iloc[start:end+1] - daily_10min['LOWER_BAND'].iloc[start:end+1]) * 0.5
            lower_band = pred_power - band_width/2
            upper_band = pred_power + band_width/2
            
            ax1.fill_between(x_pos[start:end+1], 
                           lower_band, 
                           upper_band, 
                           color='#E8F4FD', alpha=0.6, zorder=1) 
        
        # Axis setup
        setup_axes(ax1, daily_10min, use_log_scale, max_power)
        ax1.set_title(f'{date}', fontsize=20, fontweight='600', color='#f5f5f5')
        ax1.set_ylabel('Power Generation (kW)', fontsize=14, fontweight='normal', color='#aaaaaa')
        
        # Legend
        create_legend(ax1)
        
        # Save
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        plt.savefig(os.path.join(output_dir, f'anomaly_report_{date}.png'), 
                    dpi=200, bbox_inches='tight', facecolor='#181b1f', edgecolor='none', transparent=False, pad_inches=0.2)
        plt.close()


def get_consecutive_ranges(mask):
    """Return start/end indices of consecutive True ranges."""
    ranges = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ranges.append((start, i-1))
            start = None
    if start is not None:
        ranges.append((start, len(mask)-1))
    return ranges


def create_anomaly_mask(daily_10min):
    """Create anomaly mask."""
    # Error rate above 12%
    error_mask = daily_10min['ERROR_RATE'] > 12
    
    # Zero-power segments (continuous for at least 60 minutes)
    zero_power_mask = np.zeros(len(daily_10min), dtype=bool)
    zero_consecutive = 0
    
    for i, power in enumerate(daily_10min['ACTUAL_POWER']):
        if power == 0:
            zero_consecutive += 1
        else:
            if zero_consecutive >= 6:  # 10 min × 6 = 60 min
                zero_power_mask[i-zero_consecutive:i] = True
            zero_consecutive = 0
    
    if zero_consecutive >= 6:
        zero_power_mask[len(daily_10min)-zero_consecutive:] = True
    
    return error_mask | zero_power_mask


def setup_axes(ax, daily_10min, use_log_scale, max_power):
    """Configure axes."""
    # X-axis ticks every 30 minutes
    tick_indices = [i for i, t in enumerate(daily_10min['TIMESTAMP']) if t.minute % 30 == 0]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(daily_10min['TIMESTAMP'].dt.strftime('%H:%M').iloc[tick_indices], 
                       rotation=45, fontsize=13, fontweight='bold', color='#ffffff')
    
    
    # Axis ticks
    ax.tick_params(axis='both', labelsize=13, labelcolor='#ffffff', width=1, length=5, color='#555555')
    
    # Axis borders
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(1.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_legend(ax):
    """Create legend."""
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='#0066FF', alpha=1.0, label='Predicted Power'),
        mpatches.Patch(color='#FF0033', alpha=1.0, label='Actual Power'),
        mpatches.Patch(color='#E8F4FD', alpha=0.8, label='Normal Range'),
        mpatches.Patch(color='#FF8C42', alpha=0.4, label='Anomaly Detection')
    ]
    ax.legend(handles=legend_elements, fontsize=14, framealpha=0.98, 
             facecolor='#262626', edgecolor='#555555', labelcolor='#f0f0f0', 
             shadow=False, frameon=True, borderpad=0.8, loc='upper left')


def main():
    """Main function."""
    try:
        if not TEST_MODE:
            return
            
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pv_data, target_data = load_data()
        
        # Process single test date
        test_day = pd.to_datetime(TEST_DATE).normalize()
        
        # Use data only from 09:00 to 18:00
        target_data_filtered = target_data[(target_data['hour'] >= 9) & (target_data['hour'] <= 17)]
        
        # Merge data and create features
        merged = pd.merge(pv_data[['TIMESTAMP', 'sr', 'surface_temp']], 
                         target_data_filtered[['TIMESTAMP', 'power_63', 'hour']], on='TIMESTAMP', how='inner')
        merged = create_features(merged.sort_values('TIMESTAMP').reset_index(drop=True))
        
        if len(merged) == 0:
            return
        
        # Load model and run prediction
        model = CatBoostRegressor().load_model(MODEL_PATH)
        feature_columns = ['sr', 'surface_temp', 'hour', 'sin_hour', 'cos_hour', 
                          'power_lag1', 'power_lag2', 'power_lag3', 'power_per_sr']
        
        # 5-minute delayed prediction
        records = []
        current_time = test_day + pd.Timedelta(hours=9)  # start at 09:00
        end_time = test_day + pd.Timedelta(hours=18, minutes=1)  # until 18:00 of the same day
        zero_count = 0  # consecutive zero counter
        last_normal_pred = None  # latest normal prediction value
        
        while current_time <= end_time:
            target_time = current_time - pd.Timedelta(minutes=5)
            target_row = merged[merged['TIMESTAMP'] <= target_time]
            
            # Find current-time row in PV data (irradiance, temperature)
            pv_current = pv_data[pv_data['TIMESTAMP'] == current_time]
            
            if len(target_row) > 0:
                pred = float(np.abs(model.predict(target_row.iloc[[-1]][feature_columns]))[0])
                actual_row = merged[merged['TIMESTAMP'] == target_time]
                
                if len(actual_row) > 0:
                    # Case: actual value exists
                    actual, sr, hour = actual_row.iloc[0][['power_63', 'sr', 'hour']].values
                    
                    if actual == 0:
                        zero_count += 1
                        if zero_count >= 20:  
                            pred = sr * 0.005  
                    else:
                        zero_count = 0
                        last_normal_pred = pred 
                    
                    smape = calculate_smape(actual, pred)
                    is_anom = int((smape > 11) and (sr > 100) and (9 <= hour <= 16))
                    
                    records.append({
                        'TIMESTAMP': current_time, 'ACTUAL_POWER': actual, 
                        'PREDICTED_POWER': pred, 'IS_ANOMALY': is_anom, 'ERROR_RATE': smape
                    })
                
                elif len(pv_current) > 0:
                    
                    sr = pv_current.iloc[0]['sr']
                    hour = current_time.hour
                    
                    # Irradiance-based prediction (when power data is missing)
                    if sr > 50 and 9 <= hour <= 18:  
                        pred = sr * 0.008  
                    else:
                        pred = 0 
                    
                    actual = 0 
                    smape = 0  
                    is_anom = 0
                    
                    records.append({
                        'TIMESTAMP': current_time, 'ACTUAL_POWER': actual, 
                        'PREDICTED_POWER': pred, 'IS_ANOMALY': is_anom, 'ERROR_RATE': smape
                    })
            
            # Even without target_row, apply irradiance-based prediction if PV data exists
            elif len(pv_current) > 0:
                sr = pv_current.iloc[0]['sr']
                hour = current_time.hour
                
                # Irradiance-based prediction (when no generation data exists at all)
                if sr > 50 and 9 <= hour <= 18:  
                    pred = sr * 0.008  
                else:
                    pred = 0 
                
                actual = 0 
                smape = 0  
                is_anom = 0
                
                records.append({
                    'TIMESTAMP': current_time, 'ACTUAL_POWER': actual, 
                    'PREDICTED_POWER': pred, 'IS_ANOMALY': is_anom, 'ERROR_RATE': smape
                })
            
            current_time += pd.Timedelta(minutes=1)
        
        if records:
            # Process results
            sim_df = pd.DataFrame(records).set_index('TIMESTAMP')
            agg = sim_df.resample('5min').agg({
                'ACTUAL_POWER': 'mean', 'PREDICTED_POWER': 'mean',
                'IS_ANOMALY': 'max', 'ERROR_RATE': 'mean'
            }).reset_index()
            
            # Apply 60-minute continuous anomaly detection
            agg = apply_60min_continuous_anomaly_detection(agg)
            
            # Finalize results and generate plots
            result = agg[['TIMESTAMP', 'ACTUAL_POWER', 'PREDICTED_POWER', 'ERROR_RATE']].dropna().reset_index(drop=True)
            create_daily_plots(result, OUTPUT_DIR)
            
    except Exception as e:
        raise


def calculate_smape(actual, predicted):
    """Compute SMAPE."""
    if pd.isna(actual) or actual == 0:
        return 0.0
    denom = (abs(actual) + abs(predicted)) / 2
    return (abs(actual - predicted) / denom * 100) if denom != 0 else 0.0


if __name__ == "__main__":
    main()



