# 🖥️ Interactive Frontend Dashboard

## Overview
This dashboard provides a comprehensive interface for equipment failure prediction with real-time monitoring, model performance tracking, and explainability features.

## Key Components

### 1. Real-Time Failure Risk Visualization
- **Risk Gauges**: Current failure risk score (0-100%)
- **Alert System**: Color-coded warnings (Normal, Warning, Critical)
- **Live Telemetry Display**: Real-time voltage, current, temperature, and load factor readings

### 2. Model Performance Monitoring
- **Accuracy Metrics**: Precision, Recall, F1-Score, ROC-AUC tracking
- **Inference Performance**: Average and P95 latency metrics
- **Drift Detection**: Alerts when model behavior changes significantly

### 3. Historical Analysis & Trends
- **Time-Series Visualizations**: Failure predictions over time
- **Risk Score Distribution**: Understanding prediction confidence
- **Performance Trends**: Model accuracy evolution

### 4. Manual Prediction Interface
Users can submit custom telemetry data for immediate failure risk assessment with confidence scoring.

### 5. Model Explainability
- **Physics vs ML Contributions**: Shows how each model component contributes to predictions
- **Feature Importance**: Identifies which factors drive failure risk
- **Confidence Metrics**: Model agreement scoring

## Data Flow
1. User submits telemetry (voltage, current, temperature, load_factor)
2. Feature engineering pipeline processes input
3. Ensemble prediction (Physics 40% + Gradient Boosting 60%)
4. Results displayed with confidence scores and explanations
5. Predictions logged for monitoring and drift detection