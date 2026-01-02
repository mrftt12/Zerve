import pandas as pd
import numpy as np

# Clean and preprocess the telemetry data
cleaned_data = telemetry_data.copy()

# Check for missing values
missing_before = cleaned_data.isnull().sum().sum()

# Handle missing values (if any) - forward fill then backward fill
if missing_before > 0:
    cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')

# Remove duplicate timestamps (if any)
duplicates = cleaned_data.duplicated(subset=['timestamp']).sum()
cleaned_data = cleaned_data.drop_duplicates(subset=['timestamp'], keep='first')

# Sort by timestamp
cleaned_data = cleaned_data.sort_values('timestamp').reset_index(drop=True)

# Detect and cap extreme outliers (beyond 5 standard deviations)
# This preserves anomalies while removing data errors
for col in ['voltage', 'current', 'temperature']:
    mean = cleaned_data[col].mean()
    std = cleaned_data[col].std()
    lower_bound = mean - 5 * std
    upper_bound = mean + 5 * std
    
    outliers = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
    if outliers > 0:
        cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)
        print(f"Capped {outliers} extreme outliers in {col}")

# Ensure load_factor is within valid range [0, 1]
cleaned_data['load_factor'] = cleaned_data['load_factor'].clip(0, 1)

# Calculate data quality metrics
print("=" * 60)
print("DATA CLEANING & PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\nOriginal records: {len(telemetry_data)}")
print(f"Cleaned records: {len(cleaned_data)}")
print(f"Missing values handled: {missing_before}")
print(f"Duplicate timestamps removed: {duplicates}")
print(f"\nCleaned Data Quality:")
print(f"  Time coverage: {(cleaned_data['timestamp'].max() - cleaned_data['timestamp'].min()).days} days")
print(f"  Time gaps: {(cleaned_data['timestamp'].diff() > pd.Timedelta('15min')).sum()} intervals")
print(f"  Voltage range: {cleaned_data['voltage'].min():.2f}V - {cleaned_data['voltage'].max():.2f}V")
print(f"  Current range: {cleaned_data['current'].min():.2f}A - {cleaned_data['current'].max():.2f}A")
print(f"  Temperature range: {cleaned_data['temperature'].min():.2f}°C - {cleaned_data['temperature'].max():.2f}°C")
print(f"  Load factor range: {cleaned_data['load_factor'].min():.3f} - {cleaned_data['load_factor'].max():.3f}")
print(f"\nData Types:")
print(cleaned_data.dtypes)
print(f"\nCleaned Data Summary:")
print(cleaned_data.describe())