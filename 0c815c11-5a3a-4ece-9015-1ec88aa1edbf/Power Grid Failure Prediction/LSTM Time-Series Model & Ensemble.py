import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Since tensorflow is not available, we'll create the ensemble with just physics and GB models
# This still demonstrates the hybrid approach combining physics-based and ML methods

print("=" * 70)
print("HYBRID ENSEMBLE MODEL (Physics + Gradient Boosting)")
print("=" * 70)

# ENSEMBLE: Combine physics and GB models
# Get predictions on test set
ensemble_physics_pred = y_physics_pred_proba_test
ensemble_gb_pred = y_gb_pred_proba_test

# Ensemble: Average predictions with weights (50/50 since no LSTM available)
ensemble_pred_proba = (0.4 * ensemble_physics_pred + 0.6 * ensemble_gb_pred)
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

# Evaluate ensemble
ensemble_acc = accuracy_score(y_ml_test, ensemble_pred)
ensemble_precision = precision_score(y_ml_test, ensemble_pred)
ensemble_recall = recall_score(y_ml_test, ensemble_pred)
ensemble_f1 = f1_score(y_ml_test, ensemble_pred)
ensemble_roc_auc = roc_auc_score(y_ml_test, ensemble_pred_proba)

print(f"\n🎯 ENSEMBLE CONFIGURATION")
print(f"   Physics-Based Model Weight: 40%")
print(f"   Gradient Boosting Model Weight: 60%")
print(f"   Note: LSTM excluded due to tensorflow unavailability")

print(f"\n🏆 ENSEMBLE PERFORMANCE")
print(f"   Test Accuracy: {ensemble_acc:.4f}")
print(f"   Precision: {ensemble_precision:.4f}")
print(f"   Recall: {ensemble_recall:.4f}")
print(f"   F1-Score: {ensemble_f1:.4f}")
print(f"   ROC-AUC: {ensemble_roc_auc:.4f}")

print(f"\n📊 MODEL COMPARISON SUMMARY")
print(f"   {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print(f"   {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
print(f"   {'Physics-Based':<25} {physics_test_acc:>10.4f} {physics_precision:>10.4f} {physics_recall:>10.4f} {physics_f1:>10.4f} {physics_roc_auc:>10.4f}")
print(f"   {'Gradient Boosting':<25} {gb_test_acc:>10.4f} {gb_precision:>10.4f} {gb_recall:>10.4f} {gb_f1:>10.4f} {gb_roc_auc:>10.4f}")
print(f"   {'Hybrid Ensemble':<25} {ensemble_acc:>10.4f} {ensemble_precision:>10.4f} {ensemble_recall:>10.4f} {ensemble_f1:>10.4f} {ensemble_roc_auc:>10.4f}")

print(f"\n✅ KEY FINDINGS")
print(f"   • Physics-based model leverages domain knowledge (electrical engineering)")
print(f"   • Gradient Boosting captures complex patterns in 70 features")
print(f"   • Ensemble combines interpretability of physics with ML pattern recognition")
print(f"   • High recall ({ensemble_recall:.2%}) minimizes missed failure predictions")
print(f"   • ROC-AUC of {ensemble_roc_auc:.4f} shows excellent discrimination")

print("\n" + "=" * 70)

# Visualization: Model Performance Comparison
fig_comparison, ax_comp = plt.subplots(figsize=(10, 6))
fig_comparison.patch.set_facecolor('#1D1D20')
ax_comp.set_facecolor('#1D1D20')

model_names = ['Physics-Based', 'Gradient\nBoosting', 'Hybrid\nEnsemble']
metrics = {
    'Accuracy': [physics_test_acc, gb_test_acc, ensemble_acc],
    'F1-Score': [physics_f1, gb_f1, ensemble_f1],
    'ROC-AUC': [physics_roc_auc, gb_roc_auc, ensemble_roc_auc]
}

x_pos = np.arange(len(model_names))
width = 0.25
colors_metrics = ['#A1C9F4', '#FFB482', '#8DE5A1']

for _idx_metric, (metric_name, metric_values) in enumerate(metrics.items()):
    ax_comp.bar(x_pos + _idx_metric*width, metric_values, width, label=metric_name, color=colors_metrics[_idx_metric])

ax_comp.set_xlabel('Model', fontsize=12, color='#fbfbff', fontweight='bold')
ax_comp.set_ylabel('Score', fontsize=12, color='#fbfbff', fontweight='bold')
ax_comp.set_title('Hybrid Modeling: Performance Comparison', fontsize=14, color='#fbfbff', fontweight='bold', pad=15)
ax_comp.set_xticks(x_pos + width)
ax_comp.set_xticklabels(model_names, fontsize=10, color='#fbfbff')
ax_comp.tick_params(colors='#fbfbff')
ax_comp.legend(loc='lower right', framealpha=0.9, facecolor='#1D1D20', edgecolor='#909094', 
          labelcolor='#fbfbff')
ax_comp.spines['bottom'].set_color('#909094')
ax_comp.spines['left'].set_color('#909094')
ax_comp.spines['top'].set_visible(False)
ax_comp.spines['right'].set_visible(False)
ax_comp.set_ylim([0.94, 1.0])
ax_comp.grid(axis='y', alpha=0.2, color='#909094')

plt.tight_layout()
model_comparison_plot = fig_comparison
