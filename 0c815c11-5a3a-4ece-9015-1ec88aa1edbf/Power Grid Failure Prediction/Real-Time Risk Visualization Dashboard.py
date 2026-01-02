import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
import time

# REAL-TIME FAILURE RISK VISUALIZATION DASHBOARD
# Displays current risk assessment with visual gauges and alerts

print("=" * 80)
print("REAL-TIME FAILURE RISK VISUALIZATION DASHBOARD")
print("=" * 80)

# Zerve Design System Colors
BG_COLOR = '#1D1D20'
TEXT_PRIMARY = '#fbfbff'
TEXT_SECONDARY = '#909094'
HIGHLIGHT = '#ffd400'
SUCCESS = '#17b26a'
WARNING = '#f04438'
COLORS_DASH = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']

# Simulate current system state (using test example)
current_telemetry_dash = {
    'voltage': 242.5,
    'current': 35.2,
    'temperature': 48.3,
    'load_factor': 0.75,
    'timestamp': datetime.now()
}

# Get real-time prediction
current_prediction_dash = predict_failure_realtime(current_telemetry_dash)

risk_score_dash = current_prediction_dash['risk_score'] * 100  # Convert to percentage
confidence_dash = current_prediction_dash['confidence']
physics_contribution_dash = current_prediction_dash['physics_score'] * 100
ml_contribution_dash = current_prediction_dash['ml_score'] * 100
prediction_status_dash = current_prediction_dash['prediction']

# Determine alert level
if risk_score_dash < 30:
    alert_level_dash = 'NORMAL'
    alert_color_dash = SUCCESS
elif risk_score_dash < 60:
    alert_level_dash = 'WARNING'
    alert_color_dash = HIGHLIGHT
else:
    alert_level_dash = 'CRITICAL'
    alert_color_dash = WARNING

print(f"\n🎯 CURRENT SYSTEM STATUS")
print(f"   Timestamp: {current_telemetry_dash['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Alert Level: {alert_level_dash}")
print(f"   Risk Score: {risk_score_dash:.1f}%")
print(f"   Confidence: {confidence_dash:.1f}%")

# Create comprehensive dashboard visualization
dashboard_fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
dashboard_fig.suptitle('Equipment Failure Risk - Real-Time Dashboard', 
                       fontsize=18, color=TEXT_PRIMARY, y=0.98, weight='bold')

# 1. Risk Gauge (Top Left)
ax_gauge = plt.subplot(2, 3, 1, facecolor=BG_COLOR)
ax_gauge.set_xlim(0, 1)
ax_gauge.set_ylim(0, 1)
ax_gauge.axis('off')

# Draw gauge arc
theta_dash = np.linspace(np.pi, 0, 100)
radius_dash = 0.35
center_x_dash, center_y_dash = 0.5, 0.3

# Background arc
ax_gauge.plot(center_x_dash + radius_dash * np.cos(theta_dash), 
              center_y_dash + radius_dash * np.sin(theta_dash), 
              color=TEXT_SECONDARY, linewidth=20, alpha=0.3)

# Risk arc
risk_theta_dash = np.linspace(np.pi, np.pi * (1 - risk_score_dash/100), 100)
ax_gauge.plot(center_x_dash + radius_dash * np.cos(risk_theta_dash), 
              center_y_dash + radius_dash * np.sin(risk_theta_dash), 
              color=alert_color_dash, linewidth=20, alpha=0.9)

# Risk score text
ax_gauge.text(0.5, 0.35, f'{risk_score_dash:.1f}%', 
              ha='center', va='center', fontsize=32, 
              color=alert_color_dash, weight='bold')
ax_gauge.text(0.5, 0.22, 'Failure Risk', 
              ha='center', va='center', fontsize=12, color=TEXT_SECONDARY)
ax_gauge.text(0.5, 0.88, f'Status: {alert_level_dash}', 
              ha='center', va='center', fontsize=14, 
              color=alert_color_dash, weight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor=BG_COLOR, 
                       edgecolor=alert_color_dash, linewidth=2))

# 2. Live Telemetry Display (Top Middle)
ax_telemetry = plt.subplot(2, 3, 2, facecolor=BG_COLOR)
ax_telemetry.axis('off')
ax_telemetry.set_xlim(0, 1)
ax_telemetry.set_ylim(0, 1)

telemetry_items_dash = [
    ('Voltage', current_telemetry_dash['voltage'], 'V', 240, COLORS_DASH[0]),
    ('Current', current_telemetry_dash['current'], 'A', 30, COLORS_DASH[1]),
    ('Temperature', current_telemetry_dash['temperature'], '°C', 40, COLORS_DASH[2]),
    ('Load Factor', current_telemetry_dash['load_factor'], '', 0.75, COLORS_DASH[3])
]

ax_telemetry.text(0.5, 0.95, 'Live Telemetry', ha='center', va='top', 
                 fontsize=14, color=TEXT_PRIMARY, weight='bold')

for i, (label, value, unit, nominal, color) in enumerate(telemetry_items_dash):
    y_pos = 0.78 - i * 0.18
    
    # Label
    ax_telemetry.text(0.05, y_pos, label, ha='left', va='center', 
                     fontsize=11, color=TEXT_SECONDARY)
    
    # Value with unit
    value_str = f'{value:.1f}' if isinstance(value, float) and value < 10 else f'{value:.2f}'
    ax_telemetry.text(0.95, y_pos, f'{value_str} {unit}', ha='right', va='center', 
                     fontsize=12, color=color, weight='bold')
    
    # Progress bar showing deviation from nominal
    bar_y = y_pos - 0.05
    bar_width = 0.8
    bar_x_start = 0.1
    
    # Background bar
    ax_telemetry.add_patch(mpatches.Rectangle((bar_x_start, bar_y), bar_width, 0.02, 
                                              facecolor=TEXT_SECONDARY, alpha=0.3))
    
    # Value bar
    if nominal > 0:
        value_pct = min(value / (nominal * 2), 1.0)
        ax_telemetry.add_patch(mpatches.Rectangle((bar_x_start, bar_y), 
                                                  bar_width * value_pct, 0.02, 
                                                  facecolor=color, alpha=0.8))

# 3. Model Contribution Breakdown (Top Right)
ax_contrib = plt.subplot(2, 3, 3, facecolor=BG_COLOR)
ax_contrib.axis('off')
ax_contrib.set_xlim(0, 1)
ax_contrib.set_ylim(0, 1)

ax_contrib.text(0.5, 0.95, 'Model Contributions', ha='center', va='top', 
               fontsize=14, color=TEXT_PRIMARY, weight='bold')

contributions_dash = [
    ('Physics Model', physics_contribution_dash, COLORS_DASH[0]),
    ('ML Model', ml_contribution_dash, COLORS_DASH[1]),
    ('Ensemble', risk_score_dash, alert_color_dash)
]

for i, (model_name, score, color) in enumerate(contributions_dash):
    y_pos = 0.75 - i * 0.23
    
    # Model name
    ax_contrib.text(0.05, y_pos + 0.05, model_name, ha='left', va='center', 
                   fontsize=11, color=TEXT_SECONDARY)
    
    # Score
    ax_contrib.text(0.95, y_pos + 0.05, f'{score:.1f}%', ha='right', va='center', 
                   fontsize=12, color=color, weight='bold')
    
    # Horizontal bar
    bar_width_contrib = score / 100 * 0.85
    ax_contrib.add_patch(mpatches.Rectangle((0.05, y_pos - 0.03), bar_width_contrib, 0.04, 
                                           facecolor=color, alpha=0.7))
    ax_contrib.add_patch(mpatches.Rectangle((0.05, y_pos - 0.03), 0.85, 0.04, 
                                           facecolor=TEXT_SECONDARY, alpha=0.2))

# Add confidence indicator
ax_contrib.text(0.5, 0.08, f'Confidence: {confidence_dash:.1f}%', 
               ha='center', va='center', fontsize=12, color=TEXT_PRIMARY,
               bbox=dict(boxstyle='round,pad=0.7', facecolor=BG_COLOR, 
                        edgecolor=SUCCESS if confidence_dash > 80 else HIGHLIGHT, linewidth=2))

# 4. Risk Distribution History (Bottom Left)
ax_risk_dist = plt.subplot(2, 3, 4, facecolor=BG_COLOR)

# Get recent risk scores
recent_risk_scores_dash = np.array(list(model_monitor.risk_scores)) * 100 if model_monitor.risk_scores else np.array([risk_score_dash])

ax_risk_dist.hist(recent_risk_scores_dash, bins=20, color=COLORS_DASH[0], alpha=0.7, edgecolor=TEXT_PRIMARY)
ax_risk_dist.axvline(risk_score_dash, color=alert_color_dash, linestyle='--', linewidth=2, label='Current')
ax_risk_dist.set_xlabel('Risk Score (%)', fontsize=11, color=TEXT_PRIMARY)
ax_risk_dist.set_ylabel('Frequency', fontsize=11, color=TEXT_PRIMARY)
ax_risk_dist.set_title('Risk Score Distribution', fontsize=12, color=TEXT_PRIMARY, weight='bold', pad=10)
ax_risk_dist.tick_params(colors=TEXT_SECONDARY, labelsize=9)
ax_risk_dist.legend(frameon=False, labelcolor=TEXT_PRIMARY)
ax_risk_dist.spines['bottom'].set_color(TEXT_SECONDARY)
ax_risk_dist.spines['left'].set_color(TEXT_SECONDARY)
ax_risk_dist.spines['top'].set_visible(False)
ax_risk_dist.spines['right'].set_visible(False)
ax_risk_dist.set_facecolor(BG_COLOR)

# 5. Performance Metrics (Bottom Middle)
ax_metrics = plt.subplot(2, 3, 5, facecolor=BG_COLOR)
ax_metrics.axis('off')
ax_metrics.set_xlim(0, 1)
ax_metrics.set_ylim(0, 1)

ax_metrics.text(0.5, 0.95, 'Model Performance', ha='center', va='top', 
               fontsize=14, color=TEXT_PRIMARY, weight='bold')

current_performance_dash = current_metrics

performance_items_dash = [
    ('Total Predictions', f"{current_performance_dash['total_predictions']}", TEXT_PRIMARY),
    ('Avg Risk Score', f"{current_performance_dash['avg_risk_score']:.3f}", COLORS_DASH[0]),
    ('Avg Confidence', f"{current_performance_dash['avg_confidence']:.1f}%", COLORS_DASH[1]),
    ('Avg Inference Time', f"{current_performance_dash['avg_inference_time_ms']:.1f}ms", COLORS_DASH[2]),
    ('Accuracy', f"{current_performance_dash.get('accuracy', 0):.3f}", SUCCESS if current_performance_dash.get('accuracy', 0) > 0.95 else HIGHLIGHT)
]

for i, (metric, value, color) in enumerate(performance_items_dash):
    y_pos = 0.75 - i * 0.15
    ax_metrics.text(0.05, y_pos, metric, ha='left', va='center', 
                   fontsize=10, color=TEXT_SECONDARY)
    ax_metrics.text(0.95, y_pos, value, ha='right', va='center', 
                   fontsize=11, color=color, weight='bold')

# 6. Alerts & Recommendations (Bottom Right)
ax_alerts = plt.subplot(2, 3, 6, facecolor=BG_COLOR)
ax_alerts.axis('off')
ax_alerts.set_xlim(0, 1)
ax_alerts.set_ylim(0, 1)

ax_alerts.text(0.5, 0.95, 'Alerts & Recommendations', ha='center', va='top', 
              fontsize=14, color=TEXT_PRIMARY, weight='bold')

# Generate alerts based on telemetry
alerts_dash = []
if current_telemetry_dash['temperature'] > 50:
    alerts_dash.append(('⚠️', 'High temperature detected', WARNING))
if current_telemetry_dash['voltage'] > 250 or current_telemetry_dash['voltage'] < 230:
    alerts_dash.append(('⚠️', 'Voltage outside normal range', HIGHLIGHT))
if risk_score_dash > 60:
    alerts_dash.append(('🚨', 'CRITICAL: Immediate action required', WARNING))
elif risk_score_dash > 30:
    alerts_dash.append(('⚠️', 'WARNING: Monitor closely', HIGHLIGHT))
else:
    alerts_dash.append(('✅', 'System operating normally', SUCCESS))

if confidence_dash < 70:
    alerts_dash.append(('ℹ️', 'Low confidence - verify readings', COLORS_DASH[4]))

for i, (icon, message, color) in enumerate(alerts_dash[:5]):  # Max 5 alerts
    y_pos = 0.80 - i * 0.15
    ax_alerts.text(0.05, y_pos, icon, ha='left', va='center', 
                  fontsize=14, color=color)
    ax_alerts.text(0.15, y_pos, message, ha='left', va='center', 
                  fontsize=10, color=color, wrap=True)

plt.tight_layout(rect=[0, 0, 1, 0.97])

print(f"\n📊 Dashboard Components:")
print(f"   ✓ Risk Gauge with Alert Level")
print(f"   ✓ Live Telemetry Display")
print(f"   ✓ Model Contribution Breakdown")
print(f"   ✓ Risk Distribution History")
print(f"   ✓ Performance Metrics")
print(f"   ✓ Alerts & Recommendations")

print(f"\n🎨 Design: Zerve dark theme with professional styling")
print("=" * 80)

realtime_dashboard = dashboard_fig