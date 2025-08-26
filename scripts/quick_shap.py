"""
Quick SHAP Analysis for Keylogger Detection
===========================================
A streamlined version focusing on key insights.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

def quick_shap_analysis():
    print("🚀 Quick SHAP Analysis for Keylogger Detection")
    print("="*50)
    
    # Load data
    DATASET_FOLDER = "../dataset"
    print("📊 Loading data...")
    X_test = pd.read_csv(os.path.join(DATASET_FOLDER, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATASET_FOLDER, "y_test.csv")).values.ravel()
    
    # Clean column names
    X_test.columns = X_test.columns.str.strip()
    
    # Sample for efficiency
    sample_size = 200
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test[sample_indices]
    
    print(f"✅ Using {sample_size} samples for analysis")
    print(f"   Features: {X_sample.shape[1]}")
    print(f"   Classes: {len(np.unique(y_test))}")
    
    # Load CatBoost model
    print("\n🔄 Loading CatBoost model...")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(os.path.join(DATASET_FOLDER, "catboost_model.cbm"))
    print("✅ CatBoost model loaded")
    
    # SHAP analysis
    print("\n🔍 Computing SHAP values...")
    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"   Raw SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"   Number of classes: {len(shap_values)}")
        print(f"   Shape per class: {[sv.shape for sv in shap_values]}")
    else:
        print(f"   SHAP values shape: {shap_values.shape}")
    
    # For multi-class, use positive class or aggregate
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values_plot = shap_values[1]  # Binary: use positive class
            print("   Using positive class (index 1) for analysis")
        else:
            # Multi-class: aggregate all classes
            shap_values_plot = np.stack(shap_values, axis=-1).mean(axis=-1)
            print(f"   Multi-class: averaged across {len(shap_values)} classes")
    else:
        shap_values_plot = shap_values
        print("   Using single SHAP values array")
    
    print("✅ SHAP values processed")
    print(f"   Final shape for analysis: {shap_values_plot.shape}")
    
    # Feature importance
    print(f"   SHAP values shape: {shap_values_plot.shape}")
    
    # Handle different shapes of SHAP values
    if len(shap_values_plot.shape) == 2:
        # Standard case: (samples, features)
        feature_importance = np.abs(shap_values_plot).mean(0)
    else:
        # Multi-class case: take mean across all dimensions except features
        feature_importance = np.abs(shap_values_plot).mean(axis=tuple(range(len(shap_values_plot.shape)-1)))
    
    importance_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_Importance': feature_importance
    }).sort_values('SHAP_Importance', ascending=False)
    
    # Print top features
    print("\n📈 Top 15 Most Important Features:")
    print("="*50)
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:20s} | Importance: {row['SHAP_Importance']:8.4f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 8))
    top_15 = importance_df.head(15)
    plt.barh(range(len(top_15)), top_15['SHAP_Importance'])
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.xlabel('SHAP Importance')
    plt.title('Top 15 Features - SHAP Importance\nKeylogger Detection Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    results_folder = "../results"
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(f'{results_folder}/quick_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved visualization: quick_shap_analysis.png")
    
    # Save detailed results
    importance_df.to_csv(f'{results_folder}/quick_shap_results.csv', index=False)
    print(f"💾 Saved detailed results: quick_shap_results.csv")
    
    # Key insights
    print("\n🔍 Key SHAP Insights:")
    print("="*50)
    
    top_5_features = importance_df.head(5)['Feature'].tolist()
    print(f"🏆 Top 5 discriminative features:")
    for i, feature in enumerate(top_5_features, 1):
        print(f"   {i}. {feature}")
    
    print(f"\n📊 Feature categories found:")
    feature_types = {
        'Network Flow': sum(1 for f in importance_df['Feature'] if any(x in f.lower() for x in ['pkts', 'bytes', 'rate', 'dur'])),
        'Protocol': sum(1 for f in importance_df['Feature'] if 'proto_' in f),
        'State': sum(1 for f in importance_df['Feature'] if 'state_' in f),
        'Flags': sum(1 for f in importance_df['Feature'] if 'flgs_' in f),
        'Statistical': sum(1 for f in importance_df['Feature'] if any(x in f.lower() for x in ['mean', 'stddev', 'sum', 'min', 'max']))
    }
    
    for category, count in feature_types.items():
        if count > 0:
            print(f"   - {category}: {count} important features")
    
    print("\n✅ Quick SHAP Analysis Complete!")
    print("="*50)
    
    return importance_df, shap_values_plot

if __name__ == "__main__":
    importance_df, shap_values = quick_shap_analysis()
