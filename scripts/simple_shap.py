"""
Simple SHAP Analysis for Keylogger Detection
============================================
Fixed version that handles multi-class SHAP values correctly.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

def simple_shap_analysis():
    print("🚀 Simple SHAP Analysis for Keylogger Detection")
    print("="*50)
    
    # Load data
    DATASET_FOLDER = "../dataset"
    print("📊 Loading data...")
    X_test = pd.read_csv(os.path.join(DATASET_FOLDER, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATASET_FOLDER, "y_test.csv")).values.ravel()
    
    # Clean column names
    X_test.columns = X_test.columns.str.strip()
    
    # Sample for efficiency (very small for quick results)
    sample_size = 100
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
    
    print(f"   SHAP values shape: {shap_values.shape}")
    
    # For multi-class (samples, features, classes), we need to handle this correctly
    if len(shap_values.shape) == 3:
        n_samples, n_features, n_classes = shap_values.shape
        print(f"   Multi-class detected: {n_samples} samples, {n_features} features, {n_classes} classes")
        
        # Calculate feature importance as average absolute SHAP value across all samples and classes
        feature_importance = np.abs(shap_values).mean(axis=(0, 2))  # Mean across samples and classes
        print(f"   Feature importance shape: {feature_importance.shape}")
        
    elif len(shap_values.shape) == 2:
        # Binary case: (samples, features)
        print("   Binary classification detected")
        feature_importance = np.abs(shap_values).mean(axis=0)
    else:
        print(f"   Unexpected SHAP values shape: {shap_values.shape}")
        return None, None
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_Importance': feature_importance
    }).sort_values('SHAP_Importance', ascending=False)
    
    print("✅ Feature importance calculated")
    
    # Print top features
    print("\n📈 Top 15 Most Important Features:")
    print("="*50)
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:20s} | Importance: {row['SHAP_Importance']:8.4f}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    plt.figure(figsize=(12, 8))
    top_15 = importance_df.head(15)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
    bars = plt.barh(range(len(top_15)), top_15['SHAP_Importance'], color=colors)
    
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.xlabel('Average Absolute SHAP Value', fontsize=12)
    plt.title('Top 15 Most Important Features for Keylogger Detection\n(SHAP Analysis)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_15['SHAP_Importance'])):
        plt.text(value + 0.001, i, f'{value:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save results
    results_folder = "../results"
    os.makedirs(results_folder, exist_ok=True)
    
    plt.savefig(f'{results_folder}/simple_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization: simple_shap_analysis.png")
    
    # Save detailed results
    importance_df.to_csv(f'{results_folder}/simple_shap_results.csv', index=False)
    print(f"✅ Saved detailed results: simple_shap_results.csv")
    
    # Analyze feature categories
    print("\n🔍 Feature Category Analysis:")
    print("="*50)
    
    feature_categories = {
        'Flow Metrics': ['pkts', 'bytes', 'rate', 'dur'],
        'Statistics': ['mean', 'stddev', 'sum', 'min', 'max'],
        'Protocol': ['proto_'],
        'Connection State': ['state_'],
        'Network Flags': ['flgs_'],
        'Directional': ['spkts', 'dpkts', 'sbytes', 'dbytes', 'srate', 'drate'],
        'Other': ['seq', 'soui', 'doui', 'sco', 'dco']
    }
    
    category_importance = {}
    for category, keywords in feature_categories.items():
        matching_features = []
        for _, row in importance_df.iterrows():
            feature_name = row['Feature'].lower()
            if any(keyword.lower() in feature_name for keyword in keywords):
                matching_features.append(row['SHAP_Importance'])
        
        if matching_features:
            category_importance[category] = {
                'count': len(matching_features),
                'avg_importance': np.mean(matching_features),
                'total_importance': np.sum(matching_features)
            }
    
    # Sort categories by average importance
    sorted_categories = sorted(category_importance.items(), 
                              key=lambda x: x[1]['avg_importance'], reverse=True)
    
    print("📊 Feature categories by average importance:")
    for category, stats in sorted_categories:
        print(f"   {category:15s}: {stats['count']:2d} features, "
              f"Avg: {stats['avg_importance']:.4f}, "
              f"Total: {stats['total_importance']:.4f}")
    
    # Key insights
    print("\n🎯 Key Insights:")
    print("="*50)
    top_5_features = importance_df.head(5)['Feature'].tolist()
    print("🏆 Top 5 most discriminative features:")
    for i, feature in enumerate(top_5_features, 1):
        print(f"   {i}. {feature}")
    
    # Check if temporal features are important
    temporal_features = [f for f in top_5_features if any(t in f.lower() 
                        for t in ['dur', 'rate', 'time'])]
    if temporal_features:
        print(f"\n⏱️  Temporal patterns are important: {temporal_features}")
    
    # Check protocol importance
    protocol_features = [f for f in importance_df.head(10)['Feature'] 
                        if 'proto_' in f.lower()]
    if protocol_features:
        print(f"🌐 Important protocols: {protocol_features}")
    
    print("\n✅ SHAP Analysis Complete!")
    print("="*50)
    
    return importance_df, shap_values

if __name__ == "__main__":
    importance_df, shap_values = simple_shap_analysis()
