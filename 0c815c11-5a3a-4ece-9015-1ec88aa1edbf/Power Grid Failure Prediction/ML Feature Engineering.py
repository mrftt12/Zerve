import pandas as pd
import numpy as np
from scipy import stats

# Create ML and time-series features
ml_features = physics_features.copy()

# === Time-based Features ===
ml_features['hour'] = ml_features['timestamp'].dt.hour
ml_features['day_of_week'] = ml_features['timestamp'].dt.dayofweek
ml_features['day_of_month'] = ml_features['timestamp'].dt.day
ml_features['is_weekend'] = (ml_features['day_of_week'] >= 5).astype(int)

# Cyclical encoding for hour (preserves continuity between 23 and 0)
ml_features['hour_sin'] = np.sin(2 * np.pi * ml_features['hour'] / 24)
ml_features['hour_cos'] = np.cos(2 * np.pi * ml_features['hour'] / 24)

# Time of day categories
ml_features['time_of_day'] = pd.cut(ml_features['hour'], 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['night', 'morning', 'afternoon', 'evening'],
                                     include_lowest=True)
ml_features['time_of_day'] = ml_features['time_of_day'].astype(str)

# === Rolling Window Features (Time Series Patterns) ===
# 1-hour rolling statistics (4 periods at 15-min intervals)
window_1h = 4
for col in ['voltage', 'current', 'temperature', 'power_watts']:
    ml_features[f'{col}_rolling_mean_1h'] = ml_features[col].rolling(window=window_1h, min_periods=1).mean()
    ml_features[f'{col}_rolling_std_1h'] = ml_features[col].rolling(window=window_1h, min_periods=1).std()
    ml_features[f'{col}_rolling_max_1h'] = ml_features[col].rolling(window=window_1h, min_periods=1).max()
    ml_features[f'{col}_rolling_min_1h'] = ml_features[col].rolling(window=window_1h, min_periods=1).min()

# === Lag Features ===
# Previous values (useful for sequential modeling)
for col in ['voltage', 'current', 'temperature', 'load_factor']:
    ml_features[f'{col}_lag_1'] = ml_features[col].shift(1)
    ml_features[f'{col}_lag_4'] = ml_features[col].shift(4)  # 1 hour ago

# === Rate of Change Features ===
for col in ['voltage', 'current', 'temperature', 'power_watts']:
    ml_features[f'{col}_rate_of_change'] = ml_features[col].diff()
    ml_features[f'{col}_rate_of_change_pct'] = ml_features[col].pct_change() * 100

# === Anomaly Detection Features ===
# Z-score based anomaly indicators
for col in ['voltage', 'current', 'temperature']:
    mean = ml_features[col].mean()
    std = ml_features[col].std()
    ml_features[f'{col}_zscore'] = (ml_features[col] - mean) / std
    ml_features[f'{col}_is_anomaly'] = (np.abs(ml_features[f'{col}_zscore']) > 3).astype(int)

# IQR-based outlier detection
for col in ['voltage', 'current', 'temperature']:
    q1 = ml_features[col].quantile(0.25)
    q3 = ml_features[col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    ml_features[f'{col}_iqr_outlier'] = ((ml_features[col] < lower_fence) | (ml_features[col] > upper_fence)).astype(int)

# === Interaction Features ===
# Voltage-Current interaction (power relationship)
ml_features['voltage_current_interaction'] = ml_features['voltage'] * ml_features['current']

# Temperature-Load interaction
ml_features['temp_load_interaction'] = ml_features['temperature'] * ml_features['load_factor']

# === Volatility Features ===
# Rolling volatility (standard deviation)
for col in ['voltage', 'current', 'temperature']:
    ml_features[f'{col}_volatility_24h'] = ml_features[col].rolling(window=96, min_periods=1).std()

# Fill NaN values from rolling/lag operations
ml_features = ml_features.fillna(method='bfill')

print("=" * 60)
print("ML FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"\nPhysics features: {len(physics_features.columns)}")
print(f"Total features after ML engineering: {len(ml_features.columns)}")
print(f"New ML features: {len(ml_features.columns) - len(physics_features.columns)}")
print(f"\nFeature Categories:")
print(f"  - Time-based features: 9")
print(f"  - Rolling window features: 16")
print(f"  - Lag features: 8")
print(f"  - Rate of change features: 8")
print(f"  - Anomaly indicators: 9")
print(f"  - Interaction features: 2")
print(f"  - Volatility features: 3")
print(f"\nTotal anomalies detected:")
print(f"  - Voltage anomalies: {ml_features['voltage_is_anomaly'].sum()}")
print(f"  - Current anomalies: {ml_features['current_is_anomaly'].sum()}")
print(f"  - Temperature anomalies: {ml_features['temperature_is_anomaly'].sum()}")
print(f"\nDataset shape: {ml_features.shape}")
print(f"Non-null values: {ml_features.notnull().all().sum()} / {len(ml_features.columns)} columns")