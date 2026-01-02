import pandas as pd
import numpy as np

# Create physics-based features from electrical engineering principles
physics_features = cleaned_data.copy()

# === Ohm's Law and Power Features ===
# Power (Watts) = Voltage × Current
physics_features['power_watts'] = physics_features['voltage'] * physics_features['current']

# Apparent power (VA)
physics_features['apparent_power'] = physics_features['voltage'] * physics_features['current']

# Resistance (Ohms) = Voltage / Current (Ohm's Law)
# Add small epsilon to avoid division by zero
physics_features['resistance_ohms'] = physics_features['voltage'] / (physics_features['current'] + 1e-6)

# === Thermal Features ===
# Power dissipation as heat: P = I²R
physics_features['thermal_dissipation'] = (physics_features['current'] ** 2) * physics_features['resistance_ohms']

# Temperature per unit power (thermal efficiency indicator)
physics_features['temp_per_watt'] = physics_features['temperature'] / (physics_features['power_watts'] + 1)

# Thermal load index: normalized temperature relative to current
physics_features['thermal_load_index'] = physics_features['temperature'] / (physics_features['current'] + 1)

# === Thermal Degradation Indicators ===
# Temperature deviation from expected (assuming linear relationship with load)
expected_temp = 25 + (physics_features['load_factor'] * 40)  # Baseline thermal model
physics_features['temp_deviation'] = physics_features['temperature'] - expected_temp

# Thermal stress indicator (combination of high temp and high current)
# Normalized to [0, 1] range
max_temp = physics_features['temperature'].max()
max_current = physics_features['current'].max()
physics_features['thermal_stress'] = (
    (physics_features['temperature'] / max_temp) * 
    (physics_features['current'] / max_current)
)

# === Voltage Quality Features ===
# Voltage deviation from nominal (240V)
nominal_voltage = 240
physics_features['voltage_deviation'] = physics_features['voltage'] - nominal_voltage
physics_features['voltage_deviation_pct'] = (physics_features['voltage_deviation'] / nominal_voltage) * 100

# Voltage stability indicator (lower is more stable)
physics_features['voltage_instability'] = np.abs(physics_features['voltage_deviation_pct'])

# === Load and Efficiency Features ===
# Current utilization (relative to typical max)
typical_max_current = physics_features['current'].quantile(0.95)
physics_features['current_utilization'] = physics_features['current'] / typical_max_current

# System efficiency proxy (power factor approximation)
# Lower temp per watt indicates better efficiency
physics_features['efficiency_proxy'] = 1 / (physics_features['temp_per_watt'] + 0.1)

# === Combined Physics Indicators ===
# System health score (0-1, higher is better)
# Considers voltage stability, thermal stress, and efficiency
physics_features['system_health_score'] = (
    (1 - physics_features['voltage_instability'] / 100) * 0.4 +
    (1 - physics_features['thermal_stress']) * 0.4 +
    (physics_features['efficiency_proxy'] / physics_features['efficiency_proxy'].max()) * 0.2
)

print("=" * 60)
print("PHYSICS-BASED FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"\nOriginal features: {len(cleaned_data.columns)}")
print(f"Total features: {len(physics_features.columns)}")
print(f"New physics features: {len(physics_features.columns) - len(cleaned_data.columns)}")
print(f"\nNew Physics Features:")
for col in physics_features.columns:
    if col not in cleaned_data.columns:
        print(f"  - {col}: mean={physics_features[col].mean():.4f}, std={physics_features[col].std():.4f}")
print(f"\nSample of engineered features:")
print(physics_features[['timestamp', 'power_watts', 'resistance_ohms', 'thermal_dissipation', 
                        'temp_deviation', 'thermal_stress', 'system_health_score']].head(10))