import pandas as pd
import numpy as np

# Create failure target variable based on physics-based thresholds
# Failure occurs when multiple critical conditions are breached

# Define failure thresholds based on electrical engineering principles
TEMP_CRITICAL = 65  # °C - approaching thermal limits
TEMP_HIGH = 50  # °C - elevated temperature
VOLTAGE_LOW = 228  # V - 5% below nominal (240V)
VOLTAGE_HIGH = 252  # V - 5% above nominal
CURRENT_EXCESSIVE = 45  # A - approaching capacity
THERMAL_STRESS_HIGH = 0.07  # normalized thermal stress index
HEALTH_CRITICAL = 0.85  # system health score threshold

# Create binary failure indicator based on combinations of conditions
failure_conditions = pd.DataFrame()

# Critical conditions (immediate failure risk)
failure_conditions['temp_critical'] = (final_dataset['temperature'] > TEMP_CRITICAL).astype(int)
failure_conditions['voltage_critical'] = ((final_dataset['voltage'] < VOLTAGE_LOW) | 
                                          (final_dataset['voltage'] > VOLTAGE_HIGH)).astype(int)
failure_conditions['current_excessive'] = (final_dataset['current'] > CURRENT_EXCESSIVE).astype(int)
failure_conditions['thermal_stress_high'] = (final_dataset['thermal_stress'] > THERMAL_STRESS_HIGH).astype(int)
failure_conditions['health_critical'] = (final_dataset['system_health_score'] < HEALTH_CRITICAL).astype(int)

# Warning conditions (elevated risk)
failure_conditions['temp_high'] = (final_dataset['temperature'] > TEMP_HIGH).astype(int)
failure_conditions['multiple_anomalies'] = ((final_dataset['voltage_is_anomaly'] + 
                                             final_dataset['current_is_anomaly'] + 
                                             final_dataset['temperature_is_anomaly']) >= 2).astype(int)

# Failure definition: at least 2 critical conditions OR 1 critical + 2 warnings
critical_count = (failure_conditions['temp_critical'] + 
                 failure_conditions['voltage_critical'] + 
                 failure_conditions['current_excessive'] + 
                 failure_conditions['thermal_stress_high'] + 
                 failure_conditions['health_critical'])

warning_count = (failure_conditions['temp_high'] + 
                failure_conditions['multiple_anomalies'])

failure_target = ((critical_count >= 2) | ((critical_count >= 1) & (warning_count >= 2))).astype(int)

# Add target to dataset
model_data = final_dataset.copy()
model_data['failure'] = failure_target

# Remove timestamp for modeling (keep for time series later)
modeling_features = model_data.drop(['timestamp', 'time_of_day'], axis=1)

# Prepare feature sets for different models
physics_feature_cols = ['power_watts', 'apparent_power', 'resistance_ohms', 'thermal_dissipation',
                       'temp_per_watt', 'thermal_load_index', 'temp_deviation', 'thermal_stress',
                       'voltage_deviation', 'voltage_deviation_pct', 'voltage_instability',
                       'current_utilization', 'efficiency_proxy', 'system_health_score']

ml_feature_cols = [col for col in modeling_features.columns 
                   if any(x in col for x in ['rolling', 'lag', 'rate_of_change', 'zscore', 
                                              'anomaly', 'outlier', 'interaction', 'volatility'])]

base_feature_cols = ['voltage', 'current', 'temperature', 'load_factor']

time_feature_cols = ['hour', 'day_of_week', 'day_of_month', 'is_weekend', 'hour_sin', 'hour_cos']

print("=" * 70)
print("TARGET VARIABLE CREATION & DATA PREPARATION")
print("=" * 70)

print(f"\n🎯 FAILURE TARGET STATISTICS")
print(f"   Total Samples: {len(failure_target):,}")
print(f"   Failures: {failure_target.sum():,} ({failure_target.mean()*100:.2f}%)")
print(f"   Normal Operation: {(1-failure_target).sum():,} ({(1-failure_target.mean())*100:.2f}%)")
print(f"   Class Imbalance Ratio: {(1-failure_target.mean())/failure_target.mean():.2f}:1")

print(f"\n⚠️ FAILURE CONDITION BREAKDOWN")
print(f"   Temperature Critical (>{TEMP_CRITICAL}°C): {failure_conditions['temp_critical'].sum()} ({failure_conditions['temp_critical'].mean()*100:.2f}%)")
print(f"   Voltage Out of Range ({VOLTAGE_LOW}-{VOLTAGE_HIGH}V): {failure_conditions['voltage_critical'].sum()} ({failure_conditions['voltage_critical'].mean()*100:.2f}%)")
print(f"   Current Excessive (>{CURRENT_EXCESSIVE}A): {failure_conditions['current_excessive'].sum()} ({failure_conditions['current_excessive'].mean()*100:.2f}%)")
print(f"   Thermal Stress High (>{THERMAL_STRESS_HIGH}): {failure_conditions['thermal_stress_high'].sum()} ({failure_conditions['thermal_stress_high'].mean()*100:.2f}%)")
print(f"   System Health Critical (<{HEALTH_CRITICAL}): {failure_conditions['health_critical'].sum()} ({failure_conditions['health_critical'].mean()*100:.2f}%)")

print(f"\n📊 FEATURE SET SUMMARY")
print(f"   Base Features: {len(base_feature_cols)}")
print(f"   Physics Features: {len(physics_feature_cols)}")
print(f"   ML Statistical Features: {len(ml_feature_cols)}")
print(f"   Time Features: {len(time_feature_cols)}")
print(f"   Total Features: {len(modeling_features.columns) - 1} (excluding target)")

print(f"\n🔬 DATA QUALITY FOR MODELING")
print(f"   Missing Values: {modeling_features.isnull().sum().sum()}")
print(f"   Infinite Values: {np.isinf(modeling_features.select_dtypes(include=[np.number])).sum().sum()}")
print(f"   Constant Features: {(modeling_features.nunique() == 1).sum()}")

print(f"\n✅ DATASET READY FOR:")
print(f"   ✓ Physics-based Models (using {len(physics_feature_cols)} physics features)")
print(f"   ✓ ML Models - Gradient Boosting (using all {len(modeling_features.columns)-1} features)")
print(f"   ✓ ML Models - LSTM (using time-ordered sequences)")
print(f"   ✓ Ensemble Models (combining physics + ML predictions)")

print(f"\n💾 SAMPLE DATA WITH TARGET")
print(modeling_features[['voltage', 'current', 'temperature', 'thermal_stress', 
                        'system_health_score', 'failure']].head(10).to_string())

print(f"\n" + "=" * 70)
