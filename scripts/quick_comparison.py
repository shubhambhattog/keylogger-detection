"""
Quick Model Comparison - Original vs Enhanced Features
=====================================================
Fast version that runs in ~2-3 minutes with smaller models and samples.
"""

import pandas as pd
import numpy as np
import os
import time
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def quick_model_comparison():
    print("⚡ QUICK Model Comparison - Original vs Enhanced Features")
    print("="*65)
    
    start_total = time.time()
    
    # Load datasets
    print("📊 Loading datasets...")
    dataset_folder = "../dataset"
    
    X_train_orig = pd.read_csv(os.path.join(dataset_folder, "X_train.csv"))
    X_test_orig = pd.read_csv(os.path.join(dataset_folder, "X_test.csv"))
    y_train_orig = pd.read_csv(os.path.join(dataset_folder, "y_train.csv")).values.ravel()
    y_test_orig = pd.read_csv(os.path.join(dataset_folder, "y_test.csv")).values.ravel()
    
    X_train_enh = pd.read_csv(os.path.join(dataset_folder, "X_train_enhanced.csv"))
    X_test_enh = pd.read_csv(os.path.join(dataset_folder, "X_test_enhanced.csv"))
    y_train_enh = pd.read_csv(os.path.join(dataset_folder, "y_train_enhanced.csv")).values.ravel()
    y_test_enh = pd.read_csv(os.path.join(dataset_folder, "y_test_enhanced.csv")).values.ravel()
    
    print(f"   Original: {X_train_orig.shape} train, {X_test_orig.shape} test")
    print(f"   Enhanced: {X_train_enh.shape} train, {X_test_enh.shape} test")
    print(f"   New features: {X_train_enh.shape[1] - X_train_orig.shape[1]}")
    
    results = {}
    
    # Quick CatBoost comparison
    print("\n🚀 Training CatBoost models...")
    
    # Original CatBoost (fast settings)
    print("   Training original CatBoost...")
    start_time = time.time()
    cb_orig = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.2, verbose=False)
    cb_orig.fit(X_train_orig, y_train_orig)
    orig_train_time = time.time() - start_time
    
    y_pred_orig = cb_orig.predict(X_test_orig)
    y_pred_proba_orig = cb_orig.predict_proba(X_test_orig)
    
    results['CatBoost_Original'] = {
        'accuracy': accuracy_score(y_test_orig, y_pred_orig),
        'auc': roc_auc_score(y_test_orig, y_pred_proba_orig, multi_class='ovr'),
        'f1': f1_score(y_test_orig, y_pred_orig, average='weighted'),
        'train_time': orig_train_time,
        'features': X_train_orig.shape[1]
    }
    
    # Enhanced CatBoost (fast settings)
    print("   Training enhanced CatBoost...")
    start_time = time.time()
    cb_enh = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.2, verbose=False)
    cb_enh.fit(X_train_enh, y_train_enh)
    enh_train_time = time.time() - start_time
    
    y_pred_enh = cb_enh.predict(X_test_enh)
    y_pred_proba_enh = cb_enh.predict_proba(X_test_enh)
    
    results['CatBoost_Enhanced'] = {
        'accuracy': accuracy_score(y_test_enh, y_pred_enh),
        'auc': roc_auc_score(y_test_enh, y_pred_proba_enh, multi_class='ovr'),
        'f1': f1_score(y_test_enh, y_pred_enh, average='weighted'),
        'train_time': enh_train_time,
        'features': X_train_enh.shape[1]
    }
    
    # Quick LightGBM comparison
    print("\n🚀 Training LightGBM models...")
    
    # Original LightGBM (fast settings)
    print("   Training original LightGBM...")
    start_time = time.time()
    lgb_orig = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, verbose=-1)
    lgb_orig.fit(X_train_orig, y_train_orig)
    orig_train_time = time.time() - start_time
    
    y_pred_orig = lgb_orig.predict(X_test_orig)
    y_pred_proba_orig = lgb_orig.predict_proba(X_test_orig)
    
    results['LightGBM_Original'] = {
        'accuracy': accuracy_score(y_test_orig, y_pred_orig),
        'auc': roc_auc_score(y_test_orig, y_pred_proba_orig, multi_class='ovr'),
        'f1': f1_score(y_test_orig, y_pred_orig, average='weighted'),
        'train_time': orig_train_time,
        'features': X_train_orig.shape[1]
    }
    
    # Enhanced LightGBM (fast settings)
    print("   Training enhanced LightGBM...")
    start_time = time.time()
    lgb_enh = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, verbose=-1)
    lgb_enh.fit(X_train_enh, y_train_enh)
    enh_train_time = time.time() - start_time
    
    y_pred_enh = lgb_enh.predict(X_test_enh)
    y_pred_proba_enh = lgb_enh.predict_proba(X_test_enh)
    
    results['LightGBM_Enhanced'] = {
        'accuracy': accuracy_score(y_test_enh, y_pred_enh),
        'auc': roc_auc_score(y_test_enh, y_pred_proba_enh, multi_class='ovr'),
        'f1': f1_score(y_test_enh, y_pred_enh, average='weighted'),
        'train_time': enh_train_time,
        'features': X_train_enh.shape[1]
    }
    
    total_time = time.time() - start_total
    
    # Print Results
    print("\n" + "="*65)
    print("🎉 QUICK COMPARISON RESULTS")
    print("="*65)
    
    print(f"\n⏱️  Total Runtime: {total_time:.1f} seconds")
    print(f"🔢 Feature Comparison: {X_train_orig.shape[1]} → {X_train_enh.shape[1]} features")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print("-" * 65)
    print(f"{'Model':<20} {'Features':<10} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Time':<8}")
    print("-" * 65)
    
    for model_name, metrics in results.items():
        feature_type = "Original" if "Original" in model_name else "Enhanced"
        model_type = model_name.split('_')[0]
        print(f"{model_type:<20} {feature_type:<10} "
              f"{metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['train_time']:<8.2f}s")
    
    print("\n🚀 IMPROVEMENT ANALYSIS:")
    print("-" * 65)
    
    # Calculate improvements
    for model_type in ['CatBoost', 'LightGBM']:
        orig = results[f'{model_type}_Original']
        enh = results[f'{model_type}_Enhanced']
        
        acc_improvement = (enh['accuracy'] - orig['accuracy']) * 100
        auc_improvement = (enh['auc'] - orig['auc']) * 100
        f1_improvement = (enh['f1'] - orig['f1']) * 100
        
        print(f"\n{model_type} Improvements:")
        print(f"   Accuracy: {orig['accuracy']:.4f} → {enh['accuracy']:.4f} ({acc_improvement:+.2f}%)")
        print(f"   AUC:      {orig['auc']:.4f} → {enh['auc']:.4f} ({auc_improvement:+.2f}%)")
        print(f"   F1-Score: {orig['f1']:.4f} → {enh['f1']:.4f} ({f1_improvement:+.2f}%)")
    
    # Overall assessment
    avg_acc_improvement = np.mean([
        (results['CatBoost_Enhanced']['accuracy'] - results['CatBoost_Original']['accuracy']) * 100,
        (results['LightGBM_Enhanced']['accuracy'] - results['LightGBM_Original']['accuracy']) * 100
    ])
    
    print(f"\n🎯 OVERALL FEATURE ENGINEERING IMPACT:")
    print("-" * 65)
    print(f"Average Accuracy Improvement: {avg_acc_improvement:+.2f}%")
    print(f"New Features Added: {X_train_enh.shape[1] - X_train_orig.shape[1]}")
    
    if avg_acc_improvement > 0.5:
        print("✅ SUCCESS: Enhanced features significantly improved performance!")
        impact = "HIGH IMPACT"
    elif avg_acc_improvement > 0.1:
        print("✅ GOOD: Enhanced features moderately improved performance!")
        impact = "MODERATE IMPACT"
    elif avg_acc_improvement > 0:
        print("✅ MINOR: Enhanced features slightly improved performance!")
        impact = "MINOR IMPACT"
    else:
        print("⚠️ NEUTRAL: Enhanced features had minimal impact on performance.")
        impact = "MINIMAL IMPACT"
    
    # Save quick summary
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'AUC': f"{metrics['auc']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'Features': metrics['features'],
            'Training_Time': f"{metrics['train_time']:.2f}s"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("../results/quick_comparison_results.csv", index=False)
    
    print(f"\n💾 Results saved: quick_comparison_results.csv")
    print(f"🏆 Feature Engineering Impact: {impact}")
    print("="*65)
    
    return results

if __name__ == "__main__":
    results = quick_model_comparison()
