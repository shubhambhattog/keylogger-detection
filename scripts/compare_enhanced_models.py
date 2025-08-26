"""
Enhanced Model Training & Comparison
===================================
Train models with enhanced features and compare performance
against original features to measure improvement.
"""

import pandas as pd
import numpy as np
import os
import time
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelComparison:
    def __init__(self, dataset_folder="../dataset", results_folder="../results"):
        self.dataset_folder = dataset_folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)
        
    def load_datasets(self):
        """Load both original and enhanced datasets"""
        print("📊 Loading datasets...")
        
        # Original datasets
        print("   Loading original datasets...")
        X_train_orig = pd.read_csv(os.path.join(self.dataset_folder, "X_train.csv"))
        X_test_orig = pd.read_csv(os.path.join(self.dataset_folder, "X_test.csv"))
        y_train_orig = pd.read_csv(os.path.join(self.dataset_folder, "y_train.csv")).values.ravel()
        y_test_orig = pd.read_csv(os.path.join(self.dataset_folder, "y_test.csv")).values.ravel()
        
        # Enhanced datasets  
        print("   Loading enhanced datasets...")
        X_train_enh = pd.read_csv(os.path.join(self.dataset_folder, "X_train_enhanced.csv"))
        X_test_enh = pd.read_csv(os.path.join(self.dataset_folder, "X_test_enhanced.csv"))
        y_train_enh = pd.read_csv(os.path.join(self.dataset_folder, "y_train_enhanced.csv")).values.ravel()
        y_test_enh = pd.read_csv(os.path.join(self.dataset_folder, "y_test_enhanced.csv")).values.ravel()
        
        # Clean column names
        X_train_orig.columns = X_train_orig.columns.str.strip()
        X_test_orig.columns = X_test_orig.columns.str.strip()
        X_train_enh.columns = X_train_enh.columns.str.strip()
        X_test_enh.columns = X_test_enh.columns.str.strip()
        
        print(f"✅ Datasets loaded:")
        print(f"   Original: Train {X_train_orig.shape}, Test {X_test_orig.shape}")
        print(f"   Enhanced: Train {X_train_enh.shape}, Test {X_test_enh.shape}")
        print(f"   Feature improvement: {X_train_enh.shape[1] - X_train_orig.shape[1]} new features")
        
        return (X_train_orig, X_test_orig, y_train_orig, y_test_orig, 
                X_train_enh, X_test_enh, y_train_enh, y_test_enh)
    
    def train_catboost_models(self, X_train_orig, X_test_orig, y_train_orig, y_test_orig,
                             X_train_enh, X_test_enh, y_train_enh, y_test_enh):
        """Train CatBoost models on both datasets"""
        print("\n🚀 Training CatBoost Models...")
        
        results = {}
        
        # Train original model
        print("   Training CatBoost with original features...")
        start_time = time.time()
        
        catboost_orig = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=42
        )
        catboost_orig.fit(X_train_orig, y_train_orig, 
                         eval_set=(X_test_orig, y_test_orig), 
                         early_stopping_rounds=50,
                         verbose=False)
        
        orig_train_time = time.time() - start_time
        
        # Evaluate original model
        y_pred_orig = catboost_orig.predict(X_test_orig)
        y_pred_proba_orig = catboost_orig.predict_proba(X_test_orig)
        
        results['CatBoost_Original'] = {
            'model': catboost_orig,
            'accuracy': accuracy_score(y_test_orig, y_pred_orig),
            'auc': roc_auc_score(y_test_orig, y_pred_proba_orig, multi_class='ovr'),
            'precision': precision_score(y_test_orig, y_pred_orig, average='weighted'),
            'recall': recall_score(y_test_orig, y_pred_orig, average='weighted'),
            'f1': f1_score(y_test_orig, y_pred_orig, average='weighted'),
            'train_time': orig_train_time,
            'features': X_train_orig.shape[1]
        }
        
        print(f"   ✅ Original CatBoost: Accuracy={results['CatBoost_Original']['accuracy']:.4f}, "
              f"AUC={results['CatBoost_Original']['auc']:.4f}")
        
        # Train enhanced model
        print("   Training CatBoost with enhanced features...")
        start_time = time.time()
        
        catboost_enh = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=42
        )
        catboost_enh.fit(X_train_enh, y_train_enh,
                        eval_set=(X_test_enh, y_test_enh),
                        early_stopping_rounds=50,
                        verbose=False)
        
        enh_train_time = time.time() - start_time
        
        # Evaluate enhanced model
        y_pred_enh = catboost_enh.predict(X_test_enh)
        y_pred_proba_enh = catboost_enh.predict_proba(X_test_enh)
        
        results['CatBoost_Enhanced'] = {
            'model': catboost_enh,
            'accuracy': accuracy_score(y_test_enh, y_pred_enh),
            'auc': roc_auc_score(y_test_enh, y_pred_proba_enh, multi_class='ovr'),
            'precision': precision_score(y_test_enh, y_pred_enh, average='weighted'),
            'recall': recall_score(y_test_enh, y_pred_enh, average='weighted'),
            'f1': f1_score(y_test_enh, y_pred_enh, average='weighted'),
            'train_time': enh_train_time,
            'features': X_train_enh.shape[1]
        }
        
        print(f"   ✅ Enhanced CatBoost: Accuracy={results['CatBoost_Enhanced']['accuracy']:.4f}, "
              f"AUC={results['CatBoost_Enhanced']['auc']:.4f}")
        
        return results
    
    def train_lightgbm_models(self, X_train_orig, X_test_orig, y_train_orig, y_test_orig,
                             X_train_enh, X_test_enh, y_train_enh, y_test_enh, results):
        """Train LightGBM models on both datasets"""
        print("\n🚀 Training LightGBM Models...")
        
        # Train original model
        print("   Training LightGBM with original features...")
        start_time = time.time()
        
        lgb_orig = lgb.LGBMClassifier(
            objective='multiclass',
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_orig.fit(X_train_orig, y_train_orig,
                    eval_set=[(X_test_orig, y_test_orig)],
                    early_stopping_rounds=30,
                    verbose=False)
        
        orig_train_time = time.time() - start_time
        
        # Evaluate original model
        y_pred_orig = lgb_orig.predict(X_test_orig)
        y_pred_proba_orig = lgb_orig.predict_proba(X_test_orig)
        
        results['LightGBM_Original'] = {
            'model': lgb_orig,
            'accuracy': accuracy_score(y_test_orig, y_pred_orig),
            'auc': roc_auc_score(y_test_orig, y_pred_proba_orig, multi_class='ovr'),
            'precision': precision_score(y_test_orig, y_pred_orig, average='weighted'),
            'recall': recall_score(y_test_orig, y_pred_orig, average='weighted'),
            'f1': f1_score(y_test_orig, y_pred_orig, average='weighted'),
            'train_time': orig_train_time,
            'features': X_train_orig.shape[1]
        }
        
        print(f"   ✅ Original LightGBM: Accuracy={results['LightGBM_Original']['accuracy']:.4f}, "
              f"AUC={results['LightGBM_Original']['auc']:.4f}")
        
        # Train enhanced model
        print("   Training LightGBM with enhanced features...")
        start_time = time.time()
        
        lgb_enh = lgb.LGBMClassifier(
            objective='multiclass',
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_enh.fit(X_train_enh, y_train_enh,
                   eval_set=[(X_test_enh, y_test_enh)],
                   early_stopping_rounds=30,
                   verbose=False)
        
        enh_train_time = time.time() - start_time
        
        # Evaluate enhanced model
        y_pred_enh = lgb_enh.predict(X_test_enh)
        y_pred_proba_enh = lgb_enh.predict_proba(X_test_enh)
        
        results['LightGBM_Enhanced'] = {
            'model': lgb_enh,
            'accuracy': accuracy_score(y_test_enh, y_pred_enh),
            'auc': roc_auc_score(y_test_enh, y_pred_proba_enh, multi_class='ovr'),
            'precision': precision_score(y_test_enh, y_pred_enh, average='weighted'),
            'recall': recall_score(y_test_enh, y_pred_enh, average='weighted'),
            'f1': f1_score(y_test_enh, y_pred_enh, average='weighted'),
            'train_time': enh_train_time,
            'features': X_train_enh.shape[1]
        }
        
        print(f"   ✅ Enhanced LightGBM: Accuracy={results['LightGBM_Enhanced']['accuracy']:.4f}, "
              f"AUC={results['LightGBM_Enhanced']['auc']:.4f}")
        
        return results
    
    def create_comparison_visualizations(self, results):
        """Create visualizations comparing model performance"""
        print("\n📊 Creating comparison visualizations...")
        
        # Prepare data for visualization
        models = list(results.keys())
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
        
        # Create comparison dataframe
        comparison_data = []
        for model_name in models:
            model_type = model_name.split('_')[0]
            feature_type = model_name.split('_')[1]
            
            for metric in metrics:
                comparison_data.append({
                    'Model': model_type,
                    'Features': feature_type,
                    'Metric': metric.upper(),
                    'Score': results[model_name][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            metric_data = comparison_df[comparison_df['Metric'] == metric.upper()]
            
            # Create grouped bar chart
            ax = axes[i]
            sns.barplot(data=metric_data, x='Model', y='Score', hue='Features', ax=ax)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{metric.upper()} Score')
            ax.legend(title='Feature Set')
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        # Training time comparison
        train_times = [(name, results[name]['train_time']) for name in results.keys()]
        models_time = [name for name, _ in train_times]
        times = [time for _, time in train_times]
        
        ax = axes[5]
        bars = ax.bar(models_time, times, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_xticklabels(models_time, rotation=45, ha='right')
        
        # Add value labels
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/enhanced_model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: enhanced_model_comparison.png")
        
        # Create improvement summary table
        improvement_data = []
        for model_type in ['CatBoost', 'LightGBM']:
            orig_key = f'{model_type}_Original'
            enh_key = f'{model_type}_Enhanced'
            
            for metric in metrics:
                orig_score = results[orig_key][metric]
                enh_score = results[enh_key][metric]
                improvement = ((enh_score - orig_score) / orig_score) * 100
                
                improvement_data.append({
                    'Model': model_type,
                    'Metric': metric.upper(),
                    'Original': f'{orig_score:.4f}',
                    'Enhanced': f'{enh_score:.4f}',
                    'Improvement': f'{improvement:+.2f}%'
                })
        
        improvement_df = pd.DataFrame(improvement_data)
        
        # Save results
        improvement_df.to_csv(f'{self.results_folder}/model_improvement_summary.csv', index=False)
        print(f"   ✅ Saved: model_improvement_summary.csv")
        
        return improvement_df
    
    def print_summary_report(self, results, improvement_df):
        """Print comprehensive summary report"""
        print("\n" + "="*80)
        print("🎉 ENHANCED MODEL TRAINING COMPLETE!")
        print("="*80)
        
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 60)
        
        for model_name, metrics in results.items():
            model_type = model_name.replace('_', ' ')
            print(f"\n{model_type}:")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   AUC:       {metrics['auc']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1']:.4f}")
            print(f"   Features:  {metrics['features']}")
            print(f"   Train Time: {metrics['train_time']:.2f}s")
        
        print("\n🚀 IMPROVEMENT ANALYSIS:")
        print("-" * 60)
        
        # Calculate average improvements
        catboost_improvements = improvement_df[improvement_df['Model'] == 'CatBoost']
        lightgbm_improvements = improvement_df[improvement_df['Model'] == 'LightGBM']
        
        print(f"\nCatBoost Improvements (Original → Enhanced):")
        for _, row in catboost_improvements.iterrows():
            print(f"   {row['Metric']:9s}: {row['Original']} → {row['Enhanced']} ({row['Improvement']})")
        
        print(f"\nLightGBM Improvements (Original → Enhanced):")
        for _, row in lightgbm_improvements.iterrows():
            print(f"   {row['Metric']:9s}: {row['Original']} → {row['Enhanced']} ({row['Improvement']})")
        
        print("\n🎯 KEY INSIGHTS:")
        print("-" * 60)
        
        # Find best performing model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"🏆 Best Overall Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        # Calculate feature engineering impact
        catboost_acc_improvement = (results['CatBoost_Enhanced']['accuracy'] - 
                                  results['CatBoost_Original']['accuracy'])
        lightgbm_acc_improvement = (results['LightGBM_Enhanced']['accuracy'] - 
                                   results['LightGBM_Original']['accuracy'])
        
        avg_improvement = (catboost_acc_improvement + lightgbm_acc_improvement) / 2
        
        print(f"📈 Average Accuracy Improvement: {avg_improvement:+.4f} ({avg_improvement*100:+.2f}%)")
        print(f"🔧 New Features Added: {results['CatBoost_Enhanced']['features'] - results['CatBoost_Original']['features']}")
        
        if avg_improvement > 0:
            print("✅ Feature Engineering SUCCESS: Enhanced features improved performance!")
        else:
            print("⚠️ Feature Engineering had minimal impact - original features may be sufficient")
        
        print("\n📁 FILES GENERATED:")
        print("-" * 60)
        print("   • enhanced_model_comparison.png - Performance visualization")
        print("   • model_improvement_summary.csv - Detailed improvement metrics")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("🚀 Enhanced Model Training & Comparison Pipeline")
    print("="*80)
    
    # Initialize comparison class
    comparison = EnhancedModelComparison()
    
    # Load datasets
    (X_train_orig, X_test_orig, y_train_orig, y_test_orig, 
     X_train_enh, X_test_enh, y_train_enh, y_test_enh) = comparison.load_datasets()
    
    # Train CatBoost models
    results = comparison.train_catboost_models(
        X_train_orig, X_test_orig, y_train_orig, y_test_orig,
        X_train_enh, X_test_enh, y_train_enh, y_test_enh
    )
    
    # Train LightGBM models
    results = comparison.train_lightgbm_models(
        X_train_orig, X_test_orig, y_train_orig, y_test_orig,
        X_train_enh, X_test_enh, y_train_enh, y_test_enh, results
    )
    
    # Create visualizations
    improvement_df = comparison.create_comparison_visualizations(results)
    
    # Print summary report
    comparison.print_summary_report(results, improvement_df)

if __name__ == "__main__":
    main()
