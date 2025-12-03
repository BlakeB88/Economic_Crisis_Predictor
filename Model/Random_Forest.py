import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Config
INPUT_FILE = "../Preprocessing/Final Dataset/crisis_summary_columnNormalized.csv"
COUNTRY_COL = "Country"
YEAR_COL = "Year"
TARGET_COL = "target_crisis_next_3y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load the preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df[TARGET_COL].value_counts())
print(f"Crisis rate: {df[TARGET_COL].mean():.2%}")

# Seperate features from target and metadata
feature_cols = [c for c in df.columns if c not in [COUNTRY_COL, YEAR_COL, TARGET_COL, "External_Debt_Crisis"]]
X = df[feature_cols]
y = df[TARGET_COL]

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# Train-test split (stratified to maintain crisis rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training crisis rate: {y_train.mean():.2%}")
print(f"Test crisis rate: {y_test.mean():.2%}")


# Initialize Random Forest with balanced class weights
print("\n" + "="*60)
print("Training Random Forest Classifier...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,               # Number of tress
    max_depth=10,                   # Maximum tree depth
    min_samples_split=20,           # Minimum samples to split a node
    min_samples_leaf=10,            # Minimum samples in leaf node
    class_weight="balanced",        # Handle class imbalance
    random_state=RANDOM_STATE,
    n_jobs=-1                       # Use all CPU cores
)

rf_model.fit(X_train, y_train)

print("Model training complete")

# Cross-validation for robust performance estimate
print("\n" + "="*60)
print("Performing 5-Fold Stratified Cross-Validation...")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("\n--- Training Set Performance ---")
print(classification_report(y_train, y_train_pred, target_names=['No Crisis', 'Crisis']))

print("\n--- Test Set Performance ---")
print(classification_report(y_test, y_test_pred, target_names=['No Crisis', 'Crisis']))

# ROC-AUC Score
train_auc = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"\nTraining ROC-AUC: {train_auc:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")

# Feature Importance Analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_xticklabels(['No Crisis', 'Crisis'])
axes[0, 0].set_yticklabels(['No Crisis', 'Crisis'])

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(alpha=0.3)

# 3. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Prediction Probability Distribution
axes[1, 1].hist(y_test_proba[y_test == 0], bins=30, alpha=0.6, label='No Crisis', color='green')
axes[1, 1].hist(y_test_proba[y_test == 1], bins=30, alpha=0.6, label='Crisis', color='red')
axes[1, 1].set_xlabel('Predicted Probability of Crisis')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('random_forest_results.png', dpi=300, bbox_inches='tight')
print("\nVisualizations saved as 'random_forest_results.png'")

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nVisualizations saved as 'feature_importance.csv'")

# Save predictions with metadata for analysis
test_indices = X_test.index
predictions_df = pd.DataFrame({
    'Country': df.loc[test_indices, COUNTRY_COL].values,
    'Year': df.loc[test_indices, YEAR_COL].values,
    'True_Crisis': y_test.values,
    'Predicted_Crisis': y_test_pred,
    'Crisis_Probability': y_test_proba
})
predictions_df.to_csv('test_predictions.csv', index=False)
print("✓ Test predictions saved as 'test_predictions.csv'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)