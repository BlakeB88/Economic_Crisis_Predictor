import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Config
INPUT_FILE = "../Preprocessing/Final Dataset/crisis_summary_columnNormalized.csv"
COUNTRY_COL = "Country"
YEAR_COL = "Year"
TARGET_COL = "target_crisis_next_3y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load the preprocessed data
print("="*60)
print("LOADING DATA")
print("="*60)
df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df[TARGET_COL].value_counts())
print(f"Crisis rate: {df[TARGET_COL].mean():.2f}")

# Seperate features from target and metadata
feature_cols = [c for c in df.columns if c not in [COUNTRY_COL, YEAR_COL, TARGET_COL, "External_Debt_Crisis"]]
X = df[feature_cols]
y = df[TARGET_COL]

print(f"\nNumber of features: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Test crisis rate: {y_test.mean():.2%}")

# ======================
# LOGICISTIC REGRESSION
# ======================

print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL")
print("="*60)

# Train baseline logisitic regression with class balancing 
print("\nTraining Logistic Regression with L2 regularization...")
lr_model = LogisticRegression(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    max_iter=1000,
    penalty='l2',
    C=1.0
)
lr_model.fit(X_train, y_train)

# Cross-validation
print("\nPerforming 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lr_cv_score = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV ROC-AUC scores: {lr_cv_score}")
print(f"Mean CV ROC-AUC: {lr_cv_score.mean():.4f} (+/- {lr_cv_score.std() * 2:.4f})")

# Predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)
y_test_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- Logistic Regression Performance ---")
print("\nTraining Set:")
print(classification_report(y_train, y_train_pred_lr, target_names=['No Crisis', 'Crisis']))
print("\nTest Set:")
print(classification_report(y_test, y_test_pred_lr, target_names=["No Crisis", "Crisis"]))

lr_train_auc = roc_auc_score(y_train, lr_model.predict_proba(X_train)[:, 1])
lr_test_auc = roc_auc_score(y_test, y_test_proba_lr)
print(f"\nTraining ROC-AUC: {lr_train_auc:.4f}")
print(f"Test ROC-AUC: {lr_test_auc:4f}")

# Feature coefficients (importance of logisitic regression)
lr_coefficients = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\n--- Top 10 Most Important Features (by absolute coefficient) ---")
print(lr_coefficients.head(10).to_string(index=False))

# ==================
# GRADIENT BOOSITNG
# ==================

print("\n" + "="*60)
print("GRADIENT BOOSTING MODEL")
print("="*60)

# Train Gradient Boosting Classifier
print("\nTraining Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=30,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=RANDOM_STATE
)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

gb_model.fit(X_train, y_train)

# Cross-validation
print("\nPerforming 5-Fold Cross-Validation...")
gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV ROC-AUC scores: {gb_cv_scores}")
print(f"Mean CV ROC-AUC: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std() * 2:.4f})")

# Predicitions
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)
y_test_proba_gb = gb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- Gradient Boosting Perfomance ---")
print("\nTraining Set:")
print(classification_report(y_train, y_train_pred_gb, target_names=['No Crisis', 'Crisis']))
print("\nTest Set:")
print(classification_report(y_test, y_test_pred_gb, target_names=['No Crisis', 'Crisis']))

gb_train_auc = roc_auc_score(y_train, gb_model.predict_proba(X_train)[:, 1])
gb_test_auc = roc_auc_score(y_test, y_test_proba_gb)
print(f"\nTraining ROC-AUC: {gb_train_auc:.4f}")
print(f"Test ROC-AUC: {gb_test_auc:.4f}")

# Feature importance 
gb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 10 Most Important Features (Gradient Boosting) ---")
print(gb_importance.head(10).to_string(index=False))

# =================
# MODEL COMPARISON
# =================

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Gradient Boosting'],
    'Train_AUC': [lr_train_auc, gb_train_auc],
    'Test_AUC': [lr_test_auc, gb_test_auc],
    'CV_AUC_Mean': [lr_cv_score.mean(), gb_cv_scores.mean()],
    'CV_AUC_Std': [lr_cv_score.std(), gb_cv_scores.std()],
    'Overfitting_Gap': [lr_train_auc - lr_test_auc, gb_train_auc - gb_test_auc]
})

print("\n", comparison_df.to_string(index=False))

# =============================
# COMPREHENSIVE VISUALIZATIONS
# =============================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig = plt.figure(figsize=(20, 15))

# 1. Confusion Matrix - Logistic Regression
ax1 = plt.subplot(3, 4, 1)
cm_lr = confusion_matrix(y_test, y_test_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix\n(Logistic Regression)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')
ax1.set_xticklabels(['No Crisis', 'Crisis'])
ax1.set_yticklabels(['No Crisis', 'Crisis'])

# 2. Confusion Matrix - Gradient Boosting
ax2 = plt.subplot(3, 4, 2)
cm_gb = confusion_matrix(y_test, y_test_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_title('Confusion Matrix\n(Gradient Boosting)', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')
ax2.set_xticklabels(['No Crisis', 'Crisis'])
ax2.set_yticklabels(['No Crisis', 'Crisis'])

# 3. ROC Curves Comparison
ax3 = plt.subplot(3, 4, 3)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_test_proba_gb)
ax3.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'LR (AUC = {lr_test_auc:.3f})')
ax3.plot(fpr_gb, tpr_gb, color='green', lw=2, label=f'GB (AUC = {gb_test_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(alpha=0.3)

# 4. Precision-Recall Curves
ax4 = plt.subplot(3, 4, 4)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_test_proba_lr)
precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_test_proba_gb)
ap_lr = average_precision_score(y_test, y_test_proba_lr)
ap_gb = average_precision_score(y_test, y_test_proba_gb)
ax4.plot(recall_lr, precision_lr, color='blue', lw=2, label=f'LR (AP = {ap_lr:.3f})')
ax4.plot(recall_gb, precision_gb, color='green', lw=2, label=f'GB (AP = {ap_gb:.3f})')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
ax4.legend(loc="lower left")
ax4.grid(alpha=0.3)

# 5. Feature Importance - Logistic Regression (Top 15)
ax5 = plt.subplot(3, 4, 5)
top_lr = lr_coefficients.head(15)
colors = ['red' if x < 0 else 'blue' for x in top_lr['coefficient']]
ax5.barh(range(len(top_lr)), top_lr['coefficient'], color=colors, alpha=0.7)
ax5.set_yticks(range(len(top_lr)))
ax5.set_yticklabels(top_lr['feature'], fontsize=8)
ax5.invert_yaxis()
ax5.set_xlabel('Coefficient Value')
ax5.set_title('Top 15 Features\n(Logistic Regression)', fontsize=12, fontweight='bold')
ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax5.grid(axis='x', alpha=0.3)

# 6. Feature Importance - Gradient Boosting (Top 15)
ax6 = plt.subplot(3, 4, 6)
top_gb = gb_importance.head(15)
ax6.barh(range(len(top_gb)), top_gb['importance'], color='green', alpha=0.7)
ax6.set_yticks(range(len(top_gb)))
ax6.set_yticklabels(top_gb['feature'], fontsize=8)
ax6.invert_yaxis()
ax6.set_xlabel('Importance Score')
ax6.set_title('Top 15 Features\n(Gradient Boosting)', fontsize=12, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

# 7. Prediction Probability Distribution - Logistic Regression
ax7 = plt.subplot(3, 4, 7)
ax7.hist(y_test_proba_lr[y_test == 0], bins=30, alpha=0.6, label='No Crisis', color='green')
ax7.hist(y_test_proba_lr[y_test == 1], bins=30, alpha=0.6, label='Crisis', color='red')
ax7.set_xlabel('Predicted Probability')
ax7.set_ylabel('Frequency')
ax7.set_title('Probability Distribution\n(Logistic Regression)', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Prediction Probability Distribution - Gradient Boosting
ax8 = plt.subplot(3, 4, 8)
ax8.hist(y_test_proba_gb[y_test == 0], bins=30, alpha=0.6, label='No Crisis', color='green')
ax8.hist(y_test_proba_gb[y_test == 1], bins=30, alpha=0.6, label='Crisis', color='red')
ax8.set_xlabel('Predicted Probability')
ax8.set_ylabel('Frequency')
ax8.set_title('Probability Distribution\n(Gradient Boosting)', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Model Performance Comparison (AUC Scores)
ax9 = plt.subplot(3, 4, 9)
models = ['Logistic\nRegression', 'Gradient\nBoosting']
train_aucs = [lr_train_auc, gb_train_auc]
test_aucs = [lr_test_auc, gb_test_auc]
x = np.arange(len(models))
width = 0.35
ax9.bar(x - width/2, train_aucs, width, label='Train AUC', color='lightblue', edgecolor='black')
ax9.bar(x + width/2, test_aucs, width, label='Test AUC', color='lightcoral', edgecolor='black')
ax9.set_ylabel('ROC-AUC Score')
ax9.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(models)
ax9.legend()
ax9.set_ylim([0.5, 1.0])
ax9.grid(axis='y', alpha=0.3)
for i, (train, test) in enumerate(zip(train_aucs, test_aucs)):
    ax9.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', fontsize=9)
    ax9.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', fontsize=9)

# 10. Cross-Validation Scores
ax10 = plt.subplot(3, 4, 10)
ax10.boxplot([lr_cv_score, gb_cv_scores], labels=models)
ax10.set_ylabel('ROC-AUC Score')
ax10.set_title('Cross-Validation Score Distribution', fontsize=12, fontweight='bold')
ax10.grid(axis='y', alpha=0.3)

# 11. Calibration Plot - Logistic Regression
ax11 = plt.subplot(3, 4, 11)
prob_true_lr, prob_pred_lr = np.histogram(y_test_proba_lr[y_test == 1], bins=10, range=(0, 1))
prob_total_lr, _ = np.histogram(y_test_proba_lr, bins=10, range=(0, 1))
fraction_positives_lr = prob_true_lr / (prob_total_lr + 1e-10)
mean_predicted_value_lr = (np.arange(10) + 0.5) / 10
ax11.plot(mean_predicted_value_lr, fraction_positives_lr, 's-', label='Logistic Regression')
ax11.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax11.set_xlabel('Mean Predicted Probability')
ax11.set_ylabel('Fraction of Positives')
ax11.set_title('Calibration Curve\n(Logistic Regression)', fontsize=12, fontweight='bold')
ax11.legend()
ax11.grid(alpha=0.3)

# 12. Calibration Plot - Gradient Boosting
ax12 = plt.subplot(3, 4, 12)
prob_true_gb, prob_pred_gb = np.histogram(y_test_proba_gb[y_test == 1], bins=10, range=(0, 1))
prob_total_gb, _ = np.histogram(y_test_proba_gb, bins=10, range=(0, 1))
fraction_positives_gb = prob_true_gb / (prob_total_gb + 1e-10)
mean_predicted_value_gb = (np.arange(10) + 0.5) / 10
ax12.plot(mean_predicted_value_gb, fraction_positives_gb, 's-', label='Gradient Boosting', color='green')
ax12.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax12.set_xlabel('Mean Predicted Probability')
ax12.set_ylabel('Fraction of Positives')
ax12.set_title('Calibration Curve\n(Gradient Boosting)', fontsize=12, fontweight='bold')
ax12.legend()
ax12.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_gradboost_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'logistic_gradboost_results.png'")

# =============
# SAVE RESULTS
# =============

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save feature importance/coefficients
lr_coefficients.to_csv('logistic_regression_coefficients.csv', index=False)
gb_importance.to_csv('gradient_boosting_importance.csv', index=False)
print("✓ Feature importance/coefficients saved")

# Save predictions
test_indices = X_test.index
predictions_comparison = pd.DataFrame({
    'Country': df.loc[test_indices, COUNTRY_COL].values,
    'Year': df.loc[test_indices, YEAR_COL].values,
    'True_Crisis': y_test.values,
    'LR_Predicted': y_test_pred_lr,
    'LR_Probability': y_test_proba_lr,
    'GB_Predicted': y_test_pred_gb,
    'GB_Probability': y_test_proba_gb
})
predictions_comparison.to_csv('lr_gb_predictions.csv', index=False)
print("✓ Predictions saved as 'lr_gb_predictions.csv'")

# Save model comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("✓ Model comparison saved as 'model_comparison.csv'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nKey Findings:")
print(f"- Logistic Regression Test AUC: {lr_test_auc:.4f}")
print(f"- Gradient Boosting Test AUC: {gb_test_auc:.4f}")
print(f"- Best Model: {'Gradient Boosting' if gb_test_auc > lr_test_auc else 'Logistic Regression'}")
print(f"- Both models show {'minimal' if abs(lr_train_auc - lr_test_auc) < 0.05 and abs(gb_train_auc - gb_test_auc) < 0.05 else 'some'} overfitting")