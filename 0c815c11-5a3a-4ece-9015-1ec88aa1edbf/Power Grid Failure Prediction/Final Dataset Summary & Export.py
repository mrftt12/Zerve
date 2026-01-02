import pandas as pd
import numpy as np

# Create final feature-rich dataset ready for modeling
final_dataset = ml_features.copy()

# Separate feature types for documentation
original_features = ['timestamp', 'voltage', 'current', 'temperature', 'load_factor']

physics_features_list = [
    'power_watts', 'apparent_power', 'resistance_ohms', 'thermal_dissipation',
    'temp_per_watt', 'thermal_load_index', 'temp_deviation', 'thermal_stress',
    'voltage_deviation', 'voltage_deviation_pct', 'voltage_instability',
    'current_utilization', 'efficiency_proxy', 'system_health_score'
]

time_features = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend', 
    'hour_sin', 'hour_cos', 'time_of_day'
]

ml_statistical_features = [col for col in final_dataset.columns 
                           if any(x in col for x in ['rolling', 'lag', 'rate_of_change', 
                                                      'zscore', 'anomaly', 'outlier', 
                                                      'interaction', 'volatility'])]

# Summary statistics
print("=" * 70)
print("FINAL FEATURE-RICH DATASET READY FOR MODELING")
print("=" * 70)
print(f"\n📊 DATASET OVERVIEW")
print(f"   Records: {len(final_dataset):,}")
print(f"   Total Features: {len(final_dataset.columns)}")
print(f"   Time Range: {final_dataset['timestamp'].min()} to {final_dataset['timestamp'].max()}")
print(f"   Duration: {(final_dataset['timestamp'].max() - final_dataset['timestamp'].min()).days} days")
print(f"   Sampling Rate: 15 minutes")

print(f"\n🔧 FEATURE BREAKDOWN")
print(f"   Original Telemetry: {len(original_features)} features")
print(f"   Physics-Based: {len(physics_features_list)} features")
print(f"   Time-Based: {len(time_features)} features")
print(f"   ML Statistical: {len(ml_statistical_features)} features")

print(f"\n⚡ PHYSICS FEATURES (Electrical Engineering)")
for feat in physics_features_list:
    print(f"   • {feat}")

print(f"\n🤖 ML FEATURES (Time Series & Anomaly Detection)")
print(f"   • Rolling window statistics: 16 features (1h windows)")
print(f"   • Lag features: 8 features (1 & 4 periods)")
print(f"   • Rate of change: 8 features (absolute & percentage)")
print(f"   • Anomaly indicators: 9 features (z-score & IQR)")
print(f"   • Interaction features: 2 features")
print(f"   • Volatility features: 3 features (24h windows)")

print(f"\n🎯 DATA QUALITY METRICS")
print(f"   Missing Values: {final_dataset.isnull().sum().sum()}")
print(f"   Duplicate Rows: {final_dataset.duplicated().sum()}")
print(f"   Anomalies Detected:")
print(f"     - Voltage: {final_dataset['voltage_is_anomaly'].sum()} ({final_dataset['voltage_is_anomaly'].mean()*100:.2f}%)")
print(f"     - Current: {final_dataset['current_is_anomaly'].sum()} ({final_dataset['current_is_anomaly'].mean()*100:.2f}%)")
print(f"     - Temperature: {final_dataset['temperature_is_anomaly'].sum()} ({final_dataset['temperature_is_anomaly'].mean()*100:.2f}%)")

print(f"\n📈 KEY STATISTICS")
print(f"   Voltage: {final_dataset['voltage'].mean():.2f}V ± {final_dataset['voltage'].std():.2f}V")
print(f"   Current: {final_dataset['current'].mean():.2f}A ± {final_dataset['current'].std():.2f}A")
print(f"   Temperature: {final_dataset['temperature'].mean():.2f}°C ± {final_dataset['temperature'].std():.2f}°C")
print(f"   Power: {final_dataset['power_watts'].mean():.2f}W ± {final_dataset['power_watts'].std():.2f}W")
print(f"   System Health Score: {final_dataset['system_health_score'].mean():.4f} ± {final_dataset['system_health_score'].std():.4f}")

print(f"\n✅ DATASET READY FOR:")
print(f"   ✓ Predictive Maintenance Models")
print(f"   ✓ Anomaly Detection (Isolation Forest, AutoEncoder)")
print(f"   ✓ Time Series Forecasting (LSTM, Prophet)")
print(f"   ✓ Classification (Failure Prediction)")
print(f"   ✓ Regression (Load Forecasting)")

print(f"\n💾 SAMPLE DATA (First 5 rows)")
print(final_dataset[['timestamp', 'voltage', 'current', 'temperature', 'power_watts', 
                     'thermal_stress', 'system_health_score']].head().to_string())

# Create a metadata summary
metadata = {
    'total_records': len(final_dataset),
    'total_features': len(final_dataset.columns),
    'original_features': len(original_features),
    'physics_features': len(physics_features_list),
    'time_features': len(time_features),
    'ml_features': len(ml_statistical_features),
    'anomaly_rate': final_dataset[['voltage_is_anomaly', 'current_is_anomaly', 'temperature_is_anomaly']].mean().mean(),
    'avg_system_health': final_dataset['system_health_score'].mean()
}

print(f"\n" + "=" * 70)