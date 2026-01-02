import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Prepare all features for ML model (excluding target)
X_ml = modeling_features.drop('failure', axis=1)
y_ml = modeling_features['failure']

# Split data - time series aware (no shuffling)
split_idx_ml = int(len(X_ml) * 0.7)
X_ml_train, X_ml_test = X_ml[:split_idx_ml], X_ml[split_idx_ml:]
y_ml_train, y_ml_test = y_ml[:split_idx_ml], y_ml[split_idx_ml:]

# Standardize features
ml_scaler = StandardScaler()
X_ml_train_scaled = ml_scaler.fit_transform(X_ml_train)
X_ml_test_scaled = ml_scaler.transform(X_ml_test)

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_ml_train_scaled, y_ml_train)

# Make predictions
y_gb_pred_train = gb_model.predict(X_ml_train_scaled)
y_gb_pred_test = gb_model.predict(X_ml_test_scaled)
y_gb_pred_proba_test = gb_model.predict_proba(X_ml_test_scaled)[:, 1]

# Evaluate performance
gb_train_acc = accuracy_score(y_ml_train, y_gb_pred_train)
gb_test_acc = accuracy_score(y_ml_test, y_gb_pred_test)
gb_precision = precision_score(y_ml_test, y_gb_pred_test)
gb_recall = recall_score(y_ml_test, y_gb_pred_test)
gb_f1 = f1_score(y_ml_test, y_gb_pred_test)
gb_roc_auc = roc_auc_score(y_ml_test, y_gb_pred_proba_test)

# Feature importance
gb_feature_importance = pd.DataFrame({
    'feature': X_ml.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("=" * 70)
print("GRADIENT BOOSTING ML MODEL")
print("=" * 70)

print(f"\n📊 MODEL CONFIGURATION")
print(f"   Algorithm: Gradient Boosting Classifier")
print(f"   Features: {len(X_ml.columns)} (all physics + ML + time)")
print(f"   Training Samples: {len(X_ml_train):,} (70%)")
print(f"   Test Samples: {len(X_ml_test):,} (30%)")
print(f"   Estimators: 200")
print(f"   Max Depth: 5")
print(f"   Learning Rate: 0.1")

print(f"\n🎯 MODEL PERFORMANCE")
print(f"   Training Accuracy: {gb_train_acc:.4f}")
print(f"   Test Accuracy: {gb_test_acc:.4f}")
print(f"   Precision: {gb_precision:.4f}")
print(f"   Recall: {gb_recall:.4f}")
print(f"   F1-Score: {gb_f1:.4f}")
print(f"   ROC-AUC: {gb_roc_auc:.4f}")

print(f"\n🔬 TOP 10 MOST IMPORTANT FEATURES")
for _gb_idx, _gb_row in gb_feature_importance.head(10).iterrows():
    print(f"   {_gb_row['feature']:35s} → {_gb_row['importance']:.4f}")

print(f"\n📉 CONFUSION MATRIX (Test Set)")
cm_gb = confusion_matrix(y_ml_test, y_gb_pred_test)
print(f"                Predicted")
print(f"                No Fail  Fail")
print(f"   Actual  No   {cm_gb[0,0]:6d}  {cm_gb[0,1]:5d}")
print(f"           Fail {cm_gb[1,0]:6d}  {cm_gb[1,1]:5d}")

print(f"\n📋 DETAILED CLASSIFICATION REPORT")
print(classification_report(y_ml_test, y_gb_pred_test, 
                          target_names=['Normal', 'Failure']))

print("=" * 70)

# Visualization: Feature Importance
fig_gb_importance, ax_gb = plt.subplots(figsize=(10, 8))
fig_gb_importance.patch.set_facecolor('#1D1D20')
ax_gb.set_facecolor('#1D1D20')

top_features_gb = gb_feature_importance.head(15)
colors_gb = ['#A1C9F4'] * len(top_features_gb)

ax_gb.barh(range(len(top_features_gb)), top_features_gb['importance'], color=colors_gb)
ax_gb.set_yticks(range(len(top_features_gb)))
ax_gb.set_yticklabels(top_features_gb['feature'], fontsize=10, color='#fbfbff')
ax_gb.set_xlabel('Feature Importance', fontsize=12, color='#fbfbff', fontweight='bold')
ax_gb.set_title('Gradient Boosting Model: Top 15 Feature Importance', 
            fontsize=14, color='#fbfbff', fontweight='bold', pad=20)
ax_gb.tick_params(colors='#fbfbff')
ax_gb.spines['bottom'].set_color('#909094')
ax_gb.spines['left'].set_color('#909094')
ax_gb.spines['top'].set_visible(False)
ax_gb.spines['right'].set_visible(False)
ax_gb.invert_yaxis()
plt.tight_layout()

gb_importance_plot = fig_gb_importance
