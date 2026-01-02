import pandas as pd
import numpy as np

# Generate synthetic electric system telemetry data
# In production, this would load from actual data sources
np.random.seed(42)

# Time series: 30 days of data at 15-minute intervals
time_range = pd.date_range('2024-01-01', periods=2880, freq='15min')

# Generate realistic telemetry data with patterns
n_samples = len(time_range)

# Base patterns with time-of-day variation
hour_of_day = (time_range.hour + time_range.minute/60).values
daily_pattern = 0.7 + 0.3 * np.sin((hour_of_day - 6) * np.pi / 12)

# Generate voltage (V) - nominal 240V with small variations
voltage = 240 + 5 * np.sin(hour_of_day * np.pi / 12) + np.random.normal(0, 2, n_samples)

# Generate current (A) - correlated with time of day
base_current = 50 * daily_pattern
current = base_current + np.random.normal(0, 5, n_samples)
current = np.clip(current, 0, None)  # No negative current

# Generate temperature (°C) - influenced by load and ambient
ambient_temp = 25 + 10 * np.sin((hour_of_day - 14) * np.pi / 12)
load_heating = (current / 50) * 15  # Higher current increases temperature
temperature = ambient_temp + load_heating + np.random.normal(0, 2, n_samples)

# Generate load factor (0-1) - ratio of actual to peak load
peak_load = 100
actual_load = current * voltage / 1000  # Power in kW
load_factor = actual_load / peak_load
load_factor = np.clip(load_factor, 0, 1)

# Add some anomalies (5% of data)
anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
voltage[anomaly_indices] += np.random.normal(0, 15, len(anomaly_indices))
current[anomaly_indices] *= np.random.uniform(1.5, 2.5, len(anomaly_indices))
temperature[anomaly_indices] += np.random.uniform(10, 30, len(anomaly_indices))

# Create dataframe
telemetry_data = pd.DataFrame({
    'timestamp': time_range,
    'voltage': voltage,
    'current': current,
    'temperature': temperature,
    'load_factor': load_factor
})

print("=" * 60)
print("ELECTRIC SYSTEM TELEMETRY DATA LOADED")
print("=" * 60)
print(f"\nDataset Shape: {telemetry_data.shape}")
print(f"Time Range: {telemetry_data['timestamp'].min()} to {telemetry_data['timestamp'].max()}")
print(f"\nFirst 10 rows:")
print(telemetry_data.head(10).to_string())
print(f"\nData Summary Statistics:")
print(telemetry_data.describe().to_string())
print(f"\nData Types:")
print(telemetry_data.dtypes)
print(f"\nMissing Values:")
print(telemetry_data.isnull().sum())