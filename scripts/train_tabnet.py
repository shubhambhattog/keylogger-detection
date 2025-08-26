import pandas as pd
import os
import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset
DATASET_FOLDER = "../dataset"
X_train = pd.read_csv(os.path.join(DATASET_FOLDER, "X_train.csv")).values.astype(np.float32)
X_test = pd.read_csv(os.path.join(DATASET_FOLDER, "X_test.csv")).values.astype(np.float32)
y_train = pd.read_csv(os.path.join(DATASET_FOLDER, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATASET_FOLDER, "y_test.csv")).values.ravel()

# Check if GPU is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Train TabNet Model
tabnet_model = TabNetClassifier(device_name=device)
tabnet_model.fit(X_train, y_train, max_epochs=50, patience=10, batch_size=1024, virtual_batch_size=128)

# Predict and Evaluate
y_pred = tabnet_model.predict(X_test)
y_pred_proba = tabnet_model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"✅ TabNet Accuracy: {accuracy:.4f}")
print(f"✅ TabNet AUC: {auc:.4f}")

# Save the model
tabnet_model.save_model(os.path.join(DATASET_FOLDER, "tabnet_model.zip"))
