import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings

# [수정됨] XGBoost 경고 확실하게 끄기
# 메시지 패턴 매칭 대신, xgboost 모듈에서 발생하는 모든 UserWarning을 차단합니다.
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

# matplotlib 설정
#matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10


# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
METEO_PATH = os.path.join(current_dir, 'data', 'merged_meteo_d+1.csv')
TARGET_PATH = os.path.join(current_dir, 'data', 'sg0_2025-01-15_06-05.csv')
WEIGHT_DIR = os.path.join(current_dir, 'model')
RESULT_DIR = os.path.join(current_dir, 'solar_predict_result')
MODEL_PATH = os.path.join(current_dir, 'model', 'xgb_irradiance_correction_model.pkl')

os.makedirs(RESULT_DIR, exist_ok=True)


def load_meteo_data(input_date):
    """메테오 데이터 로드 및 전처리"""
    meteo_all = pd.read_csv(METEO_PATH)
    meteo_all['datetime'] = pd.to_datetime(meteo_all['datetime'])

    target_date = pd.to_datetime(input_date) + pd.Timedelta(days=1)
    meteo_all = meteo_all[meteo_all['datetime'].dt.date == target_date.date()]
    meteo_all = meteo_all[(meteo_all['hour'] >= 9) & (meteo_all['hour'] <= 18)]
    
    # XGBoost 보정 모델 로드 및 예측값 생성
    xgb = joblib.load(MODEL_PATH)
    meteo_all = meteo_all.copy()
    meteo_all['sr_sum'] = xgb.predict(meteo_all[['clearskyshortwave_instant_x60']].values)
    meteo_all['sr_sum'] = meteo_all['sr_sum'].clip(lower=0)
    
    input_df = meteo_all[['datetime', 'sr_sum']].copy()
    input_df['time'] = input_df['datetime']
    
    return input_df


def load_target_data(input_date):
    """실제 발전량 데이터 로드 및 전처리"""
    target_df = pd.read_csv(TARGET_PATH, header=0)
    
    # 60 컬럼의 차이값 계산 후 0.1 곱하여 power_60_sum 생성
    target_df['60'] = pd.to_numeric(target_df['60'], errors='coerce')
    target_df['power_60_sum'] = target_df['60'].diff() * 0.1
    target_df = target_df.dropna()
    
    # datetime 컬럼을 TIMESTAMP로 변환
    target_df['datetime_clean'] = target_df['datetime'].str.replace('_', ' ')
    target_df['datetime'] = pd.to_datetime(target_df['datetime_clean'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    target_df = target_df.dropna(subset=['datetime'])
    
    # input_date + 1일 기준으로 데이터 필터링 (예측 날짜의 발전량 데이터 사용)
    target_date = pd.to_datetime(input_date) + pd.Timedelta(days=1)
    target_df = target_df[target_df['datetime'].dt.date == target_date.date()]
    
    return target_df


def create_time_features(input_df, target_df):
    """시간 정보 컬럼 생성"""
    input_df['hour'] = input_df['time'].dt.hour
    input_df['dayofweek'] = input_df['time'].dt.dayofweek
    input_df['date'] = input_df['time'].dt.date
    target_df['hour'] = target_df['datetime'].dt.hour
    target_df['date'] = target_df['datetime'].dt.date
    
    return input_df, target_df


def get_prevday_value(df, col, time_col='time'):
    """전일 데이터 가져오기"""
    df = df.copy()
    df['prevday'] = df[time_col] - pd.Timedelta(days=1)
    prev = df[[time_col, col]].copy()
    prev[time_col] = prev[time_col] + pd.Timedelta(days=1)
    prev = prev.rename(columns={col: f'{col}_d1'})
    return pd.merge(df, prev[[time_col, f'{col}_d1']], left_on=time_col, right_on=time_col, how='left')


def create_rolling_features(input_df):
    """롤링 특성 생성"""
    grouped = input_df.groupby(input_df['time'].dt.date)
    input_df['sr_sum_rolling3'] = grouped['sr_sum'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    input_df['sr_sum_rolling6'] = grouped['sr_sum'].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    input_df['sr_sum_cumsum'] = grouped['sr_sum'].cumsum().reset_index(level=0, drop=True)
    input_df['sr_sum_diff1'] = grouped['sr_sum'].diff(1).reset_index(level=0, drop=True)
    
    for col in ['sr_sum_rolling3', 'sr_sum_rolling6', 'sr_sum_cumsum', 'sr_sum_diff1']:
        input_df[f'{col}_d1'] = input_df.groupby('hour')[col].shift(1)
    
    return input_df


def create_hour_features(input_df):
    """시간별 특성 생성"""
    input_df['sr_sum_ratio_d1'] = input_df['sr_sum'] / (input_df['sr_sum_d1'] + 1e-6)
    # target_time을 다음날 같은 시간으로 설정 (D+1 예측이므로)
    input_df['target_time'] = input_df['time'] + pd.Timedelta(days=1)
    
    # 피벗 테이블로 시간별 특성 생성
    input_df['hour_str'] = input_df['hour'].apply(lambda x: f"{x:02d}")
    pivot_df = input_df.pivot(index='date', columns='hour_str', values='sr_sum')
    pivot_df.columns = [f"sr_sum_{col}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    return input_df, pivot_df


def get_latest_weight(weight_dir, prefix):
    """최신 가중치 파일 경로 반환"""
    files = [f for f in os.listdir(weight_dir) if f.startswith(prefix) and f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f'{prefix} 파일이 없습니다.')
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(weight_dir, x)), reverse=True)
    return os.path.join(weight_dir, files[0])


def load_ensemble_models():
    """앙상블 모델 및 스케일러 로드"""
    stack_path = get_latest_weight(WEIGHT_DIR, 'ensemble_stack_')
    scaler_path = get_latest_weight(WEIGHT_DIR, 'ensemble_scaler_')
    
    stack = joblib.load(stack_path)
    
    # --------------------------------------------------------------------------
    # [FIX] LightGBM Version Compatibility Patch
    # 저장된 모델(stack) 내의 LightGBM estimators가 구버전에서 저장되어
    # _n_classes 속성이 None인 경우 발생하는 오류를 방지하기 위해 1로 강제 설정
    # --------------------------------------------------------------------------
    if hasattr(stack, 'estimators_'):
        for est in stack.estimators_:
            # LightGBM Regressor인지 확인 (속성 체크)
            if hasattr(est, '_n_classes') and est._n_classes is None:
                est._n_classes = 1
                
    # Final Estimator도 확인
    if hasattr(stack, 'final_estimator_') and stack.final_estimator_ is not None:
         est = stack.final_estimator_
         if hasattr(est, '_n_classes') and est._n_classes is None:
            est._n_classes = 1
    # --------------------------------------------------------------------------

    scaler = joblib.load(scaler_path)
    
    try:
        if hasattr(scaler, 'feature_names_in_'):
            used_features = list(scaler.feature_names_in_)
        else:
            feature_path = get_latest_weight(WEIGHT_DIR, 'ensemble_features_')
            used_features = joblib.load(feature_path)
    except Exception as e:
        raise ValueError('feature 목록을 불러올 수 없습니다: ' + str(e))
    
    return stack, scaler, used_features


def create_daily_plots(result_df, output_dir):
    for date, group in result_df.groupby(result_df['datetime'].dt.date):
        if len(group) == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(16, 6), facecolor='#181b1f')  # 8:3 비율 (16:6)  
        ax.set_facecolor('#1a1a1a')  # 더 깊은 어두운 배경
        
        # 시간 데이터 정리
        hours = group['datetime'].dt.hour.values
        pred_values = group['pred_power_60_sum'].values
        
        ax.grid(True, alpha=0.25, color='#383838', linestyle='-', linewidth=0.6, axis='y')
        ax.set_axisbelow(True)
        
        # 예측 발전량 막대그래프 - 세련된 파스텔톤
        bars = ax.bar(hours, pred_values,
                      label='Predicted Power',
                      color='#A78BFA',  # 세련된 파스텔 라벤더 (violet-400)
                      alpha=0.9,
                      width=0.6,
                      edgecolor='#555555',
                      linewidth=0.8)
        
        # 막대 위에 값 표시 - 스마트 라벨링
        max_value = max(pred_values) if len(pred_values) > 0 else 1
        for bar, value in zip(bars, pred_values):
            if value > max_value * 0.1:  # 최대값의 10% 이상인 경우만 표시
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                       f'{value:.1f}', ha='center', va='bottom',
                       fontsize=11, color='#ffffff', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 축 설정
        setup_axes(ax)
        ax.set_title(f'{date}', fontsize=20, fontweight='600', color='#f5f5f5')
        ax.set_xlabel('Time', fontsize=14, fontweight='normal', color='#aaaaaa')
        ax.set_ylabel('Power Generation (kW)', fontsize=14, fontweight='normal', color='#aaaaaa')
        
        ax.legend(fontsize=14, framealpha=0.98, loc='upper right',
                 facecolor='#262626', edgecolor='#555555', labelcolor='#f0f0f0',
                 shadow=False, frameon=True, borderpad=0.8)
        
        # 저장
        plt.tight_layout()
        fname = f'predicted_power_{date}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=200,
                    facecolor='#181b1f', edgecolor='none', transparent=False, pad_inches=0.2)
        plt.close()


def setup_axes(ax):
    """축 설정 """
    # X축 범위 설정 (9시~18시)
    ax.set_xlim(8.5, 18.5)
    ax.set_xticks(range(9, 19))
    ax.set_xticklabels([f'{h}:00' for h in range(9, 19)])
    
    # 축 눈금 -
    ax.tick_params(axis='x', labelsize=13, labelcolor='#ffffff', width=1, length=5, color='#555555')
    ax.tick_params(axis='y', labelsize=13, labelcolor='#ffffff', width=1, length=5, color='#555555')
    
    # 축 테두리 -
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(1.2)
    # 상단과 우측 테두리 숨기기
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main(input_date):
    """메인 함수"""
    try:
        # 1. 일사량 데이터 로드
        input_df = load_meteo_data(input_date)
        
        if len(input_df) == 0:
            print("일사량 데이터가 없습니다.")
            return
        
        # 2. 시간 특성 생성
        input_df['hour'] = input_df['time'].dt.hour
        input_df['dayofweek'] = input_df['time'].dt.dayofweek
        input_df['date'] = input_df['time'].dt.date
        
        # 3. 전일 데이터 추가
        input_df = get_prevday_value(input_df, 'sr_sum', 'time')
        
        # 전일 데이터가 없는 경우 기본값으로 채우기
        if 'sr_sum_d1' in input_df.columns:
            input_df['sr_sum_d1'] = input_df['sr_sum_d1'].fillna(0)
        else:
            input_df['sr_sum_d1'] = 0
        
        # power_60_sum_d1은 기본값 0으로 설정 (전일 발전량 데이터 없음)
        input_df['power_60_sum_d1'] = 0
        
        # 4. 기본 특성 컬럼 정의
        feature_cols = [
            'sr_sum', 'sr_sum_d1', 'power_60_sum_d1', 'hour', 'dayofweek'
        ]
        # 5. 롤링 특성 생성
        input_df = create_rolling_features(input_df)
        feature_cols += ['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']
        
        # 6. 시간별 특성 생성
        input_df, pivot_df = create_hour_features(input_df)
        
        # 7. 데이터 필터링
        data = input_df[(input_df['hour'] >= 9) & (input_df['hour'] <= 18)].copy()       
        
        # 9. 결측값 처리
        data[['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']] = \
            data[['sr_sum_rolling3_d1', 'sr_sum_rolling6_d1', 'sr_sum_cumsum_d1', 'sr_sum_diff1_d1']].fillna(0)
        
        # 전일 데이터 관련 결측값도 처리
        data['sr_sum_d1'] = data['sr_sum_d1'].fillna(0)
        data['power_60_sum_d1'] = data['power_60_sum_d1'].fillna(0)
        
        # 10. 피벗 테이블 병합
        if 'date' not in data.columns:
            data['date'] = data['time'].dt.date
        
        data = data.merge(pivot_df, left_on='date', right_on='date', how='left')
        
        # 11. 시간별 특성 추가
        hour_features = [f"sr_sum_{h:02d}" for h in range(9, 19)]
        feature_cols += hour_features
        data[hour_features] = data[hour_features].fillna(0)
        
        # 12. 최종 데이터 정리
        data = data.dropna(subset=feature_cols)
        
        if len(data) == 0:
            return
        
        # 13. 앙상블 모델 로드
        stack, scaler, used_features = load_ensemble_models()
        
        # 14. 예측 수행
        X = data[used_features].values
        X_scaled = scaler.transform(X)
        preds = stack.predict(X_scaled)
        
        # 15. 결과 저장
        result_df = data.copy()
        result_df['pred_power_60_sum'] = preds
        
        # 16. 저장 컬럼 정리 (예측값만 사용)
        save_cols = ['datetime', 'pred_power_60_sum']
        
        # 예측 결과의 datetime을 다음날로 설정 (D+1 예측이므로)
        if 'datetime' not in result_df.columns:
            if 'target_time' in result_df.columns:
                result_df['datetime'] = result_df['target_time']
            elif 'time' in result_df.columns:
                result_df['datetime'] = result_df['time'] + pd.Timedelta(days=1)
        
        result_df['hour'] = pd.to_datetime(result_df['datetime']).dt.hour
        
        # 17. 그래프 생성
        create_daily_plots(result_df, RESULT_DIR)
        
    except Exception as e:
        raise


if __name__ == "__main__":
    test_date = "2025-02-28"  # 원하는 날짜 입력
    main(test_date)