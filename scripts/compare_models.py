import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_curve, auc

# Load dataset
DATASET_FOLDER = "/content/drive/My Drive/dataset"
X_test_df = pd.read_csv(os.path.join(DATASET_FOLDER, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATASET_FOLDER, "y_test.csv"))['category'].values

# Prepare data for each model appropriately
# For CatBoost - DataFrame is fine
X_test_catboost = X_test_df

# For TabNet - Convert to numpy array of float32
X_test_tabnet = X_test_df.values.astype(np.float32)

# For LightGBM - DataFrame with cleaned column names
X_test_lgb = X_test_df.copy()
X_test_lgb.columns = X_test_lgb.columns.str.strip()

# Load Models
catboost_model = CatBoostClassifier()
catboost_model.load_model(os.path.join(DATASET_FOLDER, "catboost_model.cbm"))

tabnet_model = TabNetClassifier()
tabnet_model.load_model(os.path.join(DATASET_FOLDER, "tabnet_model.zip"))  # Note the double .zip extension

lgb_model = lgb.Booster(model_file=os.path.join(DATASET_FOLDER, "lightgbm_model.txt"))

# Predict Probabilities
try:
    y_pred_catboost = catboost_model.predict_proba(X_test_catboost)[:, 1]
    print("CatBoost predictions successful")
except Exception as e:
    print(f"Error with CatBoost predictions: {e}")
    y_pred_catboost = np.zeros(len(y_test))

try:
    y_pred_tabnet = tabnet_model.predict_proba(X_test_tabnet)[:, 1]
    print("TabNet predictions successful")
except Exception as e:
    print(f"Error with TabNet predictions: {e}")
    y_pred_tabnet = np.zeros(len(y_test))

try:
    y_pred_lgb = lgb_model.predict(X_test_lgb)
    print("LightGBM predictions successful")
    if len(y_pred_lgb.shape) > 1 and y_pred_lgb.shape[1] > 1:
        y_pred_lgb = y_pred_lgb[:, 1]  # Take class 1 probabilities for multi-class
except Exception as e:
    print(f"Error with LightGBM predictions: {e}")
    y_pred_lgb = np.zeros(len(y_test))

# Convert multi-class targets to binary for ROC curve (class 1 vs rest)
y_test_binary = (y_test == 1).astype(int)
print(f"Number of positive samples in test set: {np.sum(y_test_binary)}")

# Compute ROC Curves
fpr_cat, tpr_cat, _ = roc_curve(y_test_binary, y_pred_catboost)
fpr_tab, tpr_tab, _ = roc_curve(y_test_binary, y_pred_tabnet)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test_binary, y_pred_lgb)

# Compute AUC Scores
auc_cat = auc(fpr_cat, tpr_cat)
auc_tab = auc(fpr_tab, tpr_tab)
auc_lgb = auc(fpr_lgb, tpr_lgb)

# Plot ROC Curves with improved visibility
plt.figure(figsize=(10, 8))

# Plot diagonal reference line first
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')

# Plot models with different colors and styles as requested
# Include both model names AND AUC values in the legend
plt.plot(fpr_lgb, tpr_lgb, color='blue', linewidth=3, linestyle='-', label=f'LightGBM (AUC: {auc_lgb:.4f})')
plt.plot(fpr_tab, tpr_tab, color='green', linewidth=3, linestyle='-', label=f'TabNet (AUC: {auc_tab:.4f})')
plt.plot(fpr_cat, tpr_cat, color='orange', linewidth=3, linestyle='--', label=f'CatBoost (AUC: {auc_cat:.4f})')

# Add grid for better visibility
plt.grid(True, linestyle='--', alpha=0.6)

# Improve labels and title
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison of Models', fontsize=14, fontweight='bold')

# Create better legend
plt.legend(loc='lower right', fontsize=12, frameon=True, framealpha=0.9)

# Set axis limits explicitly
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

# No separate annotations in the corner of the plot

plt.tight_layout()
plt.show()