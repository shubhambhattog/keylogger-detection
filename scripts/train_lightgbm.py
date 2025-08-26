import pandas as pd
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load dataset
DATASET_FOLDER = "../dataset"
X_train = pd.read_csv(os.path.join(DATASET_FOLDER, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_FOLDER, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_FOLDER, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATASET_FOLDER, "y_test.csv")).values.ravel()

# Clean column names
X_train.columns = X_train.columns.str.strip()
X_test.columns = X_test.columns.str.strip()

# Train LightGBM Model with more conservative parameters
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    min_split_gain=0.0001,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=5,
    device='cpu'
)

# Fit model with early stopping
print("Training LightGBM model...")
lgb_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=10
)

# Predict and Evaluate
print("Making predictions...")
y_pred = lgb_model.predict(X_test)
y_pred_proba = lgb_model.predict_proba(X_test)

# Calculate metrics for multi-class
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"\n✅ LightGBM Accuracy: {accuracy:.4f}")
print(f"✅ LightGBM AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
print("Saving model...")
lgb_model.booster_.save_model(os.path.join(DATASET_FOLDER, "lightgbm_model.txt"))
print("Done!")