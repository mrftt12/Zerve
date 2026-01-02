import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Prepare physics-based features for modeling
physics_model_features = base_feature_cols + physics_feature_cols
X_physics = modeling_features[physics_model_features]
y_physics = modeling_features['failure']

# Split data - time series aware (no shuffling)
split_idx = int(len(X_physics) * 0.7)
X_physics_train, X_physics_test = X_physics[:split_idx], X_physics[split_idx:]
y_physics_train, y_physics_test = y_physics[:split_idx], y_physics[split_idx:]

# Standardize features
physics_scaler = StandardScaler()
X_physics_train_scaled = physics_scaler.fit_transform(X_physics_train)
X_physics_test_scaled = physics_scaler.transform(X_physics_test)

# Train physics-based logistic regression model
# Using physics principles to guide predictions
physics_model = LogisticRegression(
    max_iter=1000, 
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
physics_model.fit(X_physics_train_scaled, y_physics_train)

# Make predictions
y_physics_pred_train = physics_model.predict(X_physics_train_scaled)
y_physics_pred_test = physics_model.predict(X_physics_test_scaled)
y_physics_pred_proba_test = physics_model.predict_proba(X_physics_test_scaled)[:, 1]

# Evaluate performance
physics_train_acc = accuracy_score(y_physics_train, y_physics_pred_train)
physics_test_acc = accuracy_score(y_physics_test, y_physics_pred_test)
physics_precision = precision_score(y_physics_test, y_physics_pred_test)
physics_recall = recall_score(y_physics_test, y_physics_pred_test)
physics_f1 = f1_score(y_physics_test, y_physics_pred_test)
physics_roc_auc = roc_auc_score(y_physics_test, y_physics_pred_proba_test)

# Feature importance
physics_feature_importance = pd.DataFrame({
    'feature': physics_model_features,
    'coefficient': physics_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("=" * 70)
print("PHYSICS-BASED FAILURE PREDICTION MODEL")
print("=" * 70)

print(f"\n📊 MODEL CONFIGURATION")
print(f"   Algorithm: Logistic Regression (Physics-Based)")
print(f"   Features: {len(physics_model_features)} (base telemetry + physics)")
print(f"   Training Samples: {len(X_physics_train):,} (70%)")
print(f"   Test Samples: {len(X_physics_test):,} (30%)")
print(f"   Class Balance Strategy: Balanced weights")

print(f"\n🎯 MODEL PERFORMANCE")
print(f"   Training Accuracy: {physics_train_acc:.4f}")
print(f"   Test Accuracy: {physics_test_acc:.4f}")
print(f"   Precision: {physics_precision:.4f}")
print(f"   Recall: {physics_recall:.4f}")
print(f"   F1-Score: {physics_f1:.4f}")
print(f"   ROC-AUC: {physics_roc_auc:.4f}")

print(f"\n🔬 TOP 10 MOST IMPORTANT PHYSICS FEATURES")
for _idx, _row in physics_feature_importance.head(10).iterrows():
    print(f"   {_row['feature']:30s} → {_row['coefficient']:+.4f}")

print(f"\n📉 CONFUSION MATRIX (Test Set)")
cm_physics = confusion_matrix(y_physics_test, y_physics_pred_test)
print(f"                Predicted")
print(f"                No Fail  Fail")
print(f"   Actual  No   {cm_physics[0,0]:6d}  {cm_physics[0,1]:5d}")
print(f"           Fail {cm_physics[1,0]:6d}  {cm_physics[1,1]:5d}")

print(f"\n📋 DETAILED CLASSIFICATION REPORT")
print(classification_report(y_physics_test, y_physics_pred_test, 
                          target_names=['Normal', 'Failure']))

print("=" * 70)

# Visualization: Feature Importance
fig_importance, ax = plt.subplots(figsize=(10, 8))
fig_importance.patch.set_facecolor('#1D1D20')
ax.set_facecolor('#1D1D20')

top_features = physics_feature_importance.head(15)
colors = ['#A1C9F4' if x > 0 else '#FF9F9B' for x in top_features['coefficient']]

ax.barh(range(len(top_features)), top_features['coefficient'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=10, color='#fbfbff')
ax.set_xlabel('Coefficient (Impact on Failure Prediction)', fontsize=12, color='#fbfbff', fontweight='bold')
ax.set_title('Physics-Based Model: Top 15 Feature Importance', 
            fontsize=14, color='#fbfbff', fontweight='bold', pad=20)
ax.axvline(x=0, color='#909094', linestyle='--', linewidth=1)
ax.tick_params(colors='#fbfbff')
ax.spines['bottom'].set_color('#909094')
ax.spines['left'].set_color('#909094')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()

physics_importance_plot = fig_importance
