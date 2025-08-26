import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load dataset
DATASET_FOLDER = "../dataset"
X_train = pd.read_csv(os.path.join(DATASET_FOLDER, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_FOLDER, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_FOLDER, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATASET_FOLDER, "y_test.csv")).values.ravel()

# Train CatBoost Model
catboost_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=100)
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Predict and Evaluate
y_pred = catboost_model.predict(X_test)
y_pred_proba = catboost_model.predict_proba(X_test)

# Calculate metrics for a multi-class problem using One-vs-Rest strategy
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print(f"✅ CatBoost Accuracy: {accuracy:.4f}")
print(f"✅ CatBoost AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
catboost_model.save_model(os.path.join(DATASET_FOLDER, "catboost_model.cbm"))
