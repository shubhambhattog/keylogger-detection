"""
SHAP Analysis for Keylogger Detection Models
=============================================

This script provides comprehensive SHAP (SHapley Additive exPlanations) analysis
for the trained keylogger detection models. SHAP values help us understand:
1. Which features are most important for predictions
2. How each feature contributes to individual predictions
3. The relationship between feature values and model output

Author: Your Name
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SHAPAnalyzer:
    """
    A comprehensive SHAP analysis class for keylogger detection models
    """
    
    def __init__(self, dataset_folder="../dataset", results_folder="../results"):
        """
        Initialize the SHAP analyzer
        
        Args:
            dataset_folder: Path to dataset folder
            results_folder: Path to save results
        """
        self.dataset_folder = dataset_folder
        self.results_folder = results_folder
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Create results folder if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)
        
    def load_data(self):
        """Load the preprocessed dataset"""
        print("🔄 Loading dataset...")
        
        self.X_train = pd.read_csv(os.path.join(self.dataset_folder, "X_train.csv"))
        self.X_test = pd.read_csv(os.path.join(self.dataset_folder, "X_test.csv"))
        self.y_train = pd.read_csv(os.path.join(self.dataset_folder, "y_train.csv")).values.ravel()
        self.y_test = pd.read_csv(os.path.join(self.dataset_folder, "y_test.csv")).values.ravel()
        
        # Clean column names
        self.X_train.columns = self.X_train.columns.str.strip()
        self.X_test.columns = self.X_test.columns.str.strip()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        print(f"   Number of classes: {len(np.unique(self.y_train))}")
        
    def load_models(self):
        """Load the pre-trained models"""
        print("🔄 Loading pre-trained models...")
        
        # Load CatBoost model
        try:
            catboost_path = os.path.join(self.dataset_folder, "catboost_model.cbm")
            if os.path.exists(catboost_path):
                self.models['CatBoost'] = CatBoostClassifier()
                self.models['CatBoost'].load_model(catboost_path)
                print("   ✅ CatBoost model loaded")
            else:
                print("   ⚠️ CatBoost model not found")
        except Exception as e:
            print(f"   ❌ Error loading CatBoost: {e}")
        
        # Load LightGBM model
        try:
            lgb_path = os.path.join(self.dataset_folder, "lightgbm_model.txt")
            if os.path.exists(lgb_path):
                self.models['LightGBM'] = lgb.Booster(model_file=lgb_path)
                print("   ✅ LightGBM model loaded")
            else:
                print("   ⚠️ LightGBM model not found")
        except Exception as e:
            print(f"   ❌ Error loading LightGBM: {e}")
            
        print(f"✅ {len(self.models)} models loaded successfully!")
    
    def analyze_catboost_shap(self, sample_size=1000):
        """
        Perform SHAP analysis for CatBoost model
        
        Args:
            sample_size: Number of samples to use for SHAP analysis (for efficiency)
        """
        if 'CatBoost' not in self.models:
            print("❌ CatBoost model not available for SHAP analysis")
            return
            
        print("🔄 Performing SHAP analysis for CatBoost model...")
        
        model = self.models['CatBoost']
        
        # Sample data for efficiency (SHAP can be computationally expensive)
        if len(self.X_test) > sample_size:
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
            y_sample = self.y_test[sample_indices]
        else:
            X_sample = self.X_test
            y_sample = self.y_test
        
        # Create SHAP explainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For multi-class, shap_values is a list of arrays (one for each class)
        if isinstance(shap_values, list):
            # For binary classification or focusing on positive class
            if len(shap_values) == 2:
                shap_values_plot = shap_values[1]  # Positive class
                class_name = "Keylogger"
            else:
                shap_values_plot = shap_values[0]  # First class
                class_name = f"Class 0"
        else:
            shap_values_plot = shap_values
            class_name = "Prediction"
        
        # Create visualizations
        self._create_shap_visualizations(
            shap_values_plot, X_sample, model_name="CatBoost", 
            class_name=class_name, explainer=explainer
        )
        
        # Feature importance analysis
        self._analyze_feature_importance(shap_values_plot, X_sample, "CatBoost")
        
        return shap_values, explainer
    
    def analyze_lightgbm_shap(self, sample_size=1000):
        """
        Perform SHAP analysis for LightGBM model
        
        Args:
            sample_size: Number of samples to use for SHAP analysis
        """
        if 'LightGBM' not in self.models:
            print("❌ LightGBM model not available for SHAP analysis")
            return
            
        print("🔄 Performing SHAP analysis for LightGBM model...")
        
        model = self.models['LightGBM']
        
        # Sample data for efficiency
        if len(self.X_test) > sample_size:
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
            y_sample = self.y_test[sample_indices]
        else:
            X_sample = self.X_test
            y_sample = self.y_test
        
        # Create SHAP explainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values_plot = shap_values[1]  # Positive class
                class_name = "Keylogger"
            else:
                shap_values_plot = shap_values[0]  # First class
                class_name = f"Class 0"
        else:
            shap_values_plot = shap_values
            class_name = "Prediction"
        
        # Create visualizations
        self._create_shap_visualizations(
            shap_values_plot, X_sample, model_name="LightGBM", 
            class_name=class_name, explainer=explainer
        )
        
        # Feature importance analysis
        self._analyze_feature_importance(shap_values_plot, X_sample, "LightGBM")
        
        return shap_values, explainer
    
    def _create_shap_visualizations(self, shap_values, X_sample, model_name, class_name, explainer):
        """
        Create various SHAP visualizations
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            model_name: Name of the model
            class_name: Name of the class being analyzed
            explainer: SHAP explainer object
        """
        
        # 1. Summary Plot (Feature Importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'{model_name} - Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/{model_name}_shap_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {model_name}_shap_feature_importance.png")
        
        # 2. Summary Plot (Feature Effects)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'{model_name} - Feature Effects (SHAP)\nHow each feature affects {class_name} prediction', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/{model_name}_shap_summary_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {model_name}_shap_summary_plot.png")
        
        # 3. Feature Interaction Plot for top features (instead of partial dependence)
        top_features = np.argsort(np.abs(shap_values).mean(0))[-6:]  # Top 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        try:
            for i, feature_idx in enumerate(top_features):
                feature_name = X_sample.columns[feature_idx]
                
                # Create scatter plot showing feature value vs SHAP value
                axes[i].scatter(X_sample.iloc[:, feature_idx], shap_values[:, feature_idx], 
                              alpha=0.6, s=20)
                axes[i].set_xlabel(f'{feature_name}')
                axes[i].set_ylabel('SHAP Value')
                axes[i].set_title(f'{feature_name}', fontsize=10, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_name} - Feature Effects\nTop 6 Most Important Features', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/{model_name}_feature_effects.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {model_name}_feature_effects.png")
        except Exception as e:
            print(f"   ⚠️ Could not create feature effects plot: {e}")
            plt.close()
        
        # 4. Force Plot for individual predictions (sample)
        try:
            if len(shap_values) > 0:
                # Show force plot for first few samples
                for sample_idx in range(min(3, len(shap_values))):
                    shap.force_plot(
                        explainer.expected_value, shap_values[sample_idx], 
                        X_sample.iloc[sample_idx], matplotlib=True, show=False,
                        figsize=(20, 3)
                    )
                    plt.savefig(f'{self.results_folder}/{model_name}_force_plot_sample_{sample_idx}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                print(f"   ✅ Saved: {model_name}_force_plot_sample_*.png")
        except Exception as e:
            print(f"   ⚠️ Could not create force plots: {e}")
    
    def _analyze_feature_importance(self, shap_values, X_sample, model_name):
        """
        Analyze and save feature importance metrics
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            model_name: Name of the model
        """
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'SHAP_Importance': feature_importance,
            'Abs_SHAP_Importance': np.abs(feature_importance)
        }).sort_values('Abs_SHAP_Importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(f'{self.results_folder}/{model_name}_feature_importance.csv', index=False)
        print(f"   ✅ Saved: {model_name}_feature_importance.csv")
        
        # Print top 10 features
        print(f"\n📊 Top 10 Most Important Features for {model_name}:")
        print("="*50)
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:20s} | SHAP: {row['SHAP_Importance']:8.4f}")
        print("="*50)
        
        return importance_df
    
    def compare_models_shap(self):
        """
        Compare SHAP feature importance across different models
        """
        print("🔄 Comparing SHAP feature importance across models...")
        
        # Load individual feature importance files
        comparison_data = {}
        
        for model_name in self.models.keys():
            importance_file = f'{self.results_folder}/{model_name}_feature_importance.csv'
            if os.path.exists(importance_file):
                df = pd.read_csv(importance_file)
                comparison_data[model_name] = df.set_index('Feature')['Abs_SHAP_Importance']
        
        if not comparison_data:
            print("❌ No feature importance data found for comparison")
            return
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data).fillna(0)
        
        # Calculate correlation between models
        correlation_matrix = comparison_df.corr()
        
        # Visualize comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Top 15 features comparison
        top_features = comparison_df.sum(axis=1).nlargest(15).index
        comparison_subset = comparison_df.loc[top_features]
        
        comparison_subset.plot(kind='bar', ax=ax1)
        ax1.set_title('SHAP Feature Importance Comparison\nTop 15 Features', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('SHAP Importance')
        ax1.legend(title='Models')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Model Agreement on Feature Importance\nCorrelation Matrix', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/model_comparison_shap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: model_comparison_shap.png")
        
        # Save comparison data
        comparison_df.to_csv(f'{self.results_folder}/feature_importance_comparison.csv')
        print(f"   ✅ Saved: feature_importance_comparison.csv")
        
        return comparison_df
    
    def generate_shap_report(self):
        """
        Generate a comprehensive SHAP analysis report
        """
        print("📄 Generating SHAP Analysis Report...")
        
        report_content = f"""
# SHAP Analysis Report - Keylogger Detection Models
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Training samples: {len(self.X_train):,}
- Test samples: {len(self.X_test):,}
- Number of features: {self.X_train.shape[1]}
- Number of classes: {len(np.unique(self.y_train))}

## Models Analyzed
"""
        
        for model_name in self.models.keys():
            report_content += f"- {model_name}\n"
        
        report_content += """
## SHAP Analysis Results

### What SHAP Analysis Tells Us:
1. **Feature Importance**: Which network traffic features are most important for detecting keyloggers
2. **Feature Effects**: How each feature value affects the prediction (positive/negative impact)
3. **Model Interpretability**: Understanding why the model makes specific predictions
4. **Feature Interactions**: How different features work together

### Key Insights:
- The most important features for keylogger detection are identified
- We can see which network patterns are suspicious vs. normal
- The analysis helps security analysts understand model decisions
- Feature importance is consistent across different models (if multiple models agree)

### Files Generated:
- *_shap_feature_importance.png: Bar chart of most important features
- *_shap_summary_plot.png: Detailed feature effects visualization
- *_partial_dependence.png: How individual features affect predictions
- *_force_plot_sample_*.png: Explanation of individual predictions
- *_feature_importance.csv: Numerical feature importance values
- model_comparison_shap.png: Comparison across models
- feature_importance_comparison.csv: Detailed comparison data

### Next Steps:
1. Review top important features and validate with domain knowledge
2. Use insights to improve feature engineering
3. Implement model explanations in production system
4. Consider feature selection based on SHAP importance
"""
        
        # Save report
        with open(f'{self.results_folder}/shap_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        print(f"   ✅ Saved: shap_analysis_report.md")
        print("📄 SHAP Analysis Report completed!")

def main():
    """
    Main function to run comprehensive SHAP analysis
    """
    print("🚀 Starting Comprehensive SHAP Analysis for Keylogger Detection")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SHAPAnalyzer()
    
    # Load data and models
    analyzer.load_data()
    analyzer.load_models()
    
    if not analyzer.models:
        print("❌ No models found! Please train models first.")
        return
    
    # Perform SHAP analysis for each model
    if 'CatBoost' in analyzer.models:
        analyzer.analyze_catboost_shap(sample_size=500)  # Reduced sample size
        print()
    
    if 'LightGBM' in analyzer.models:
        analyzer.analyze_lightgbm_shap(sample_size=500)  # Reduced sample size
        print()
    
    # Compare models if multiple available
    if len(analyzer.models) > 1:
        analyzer.compare_models_shap()
        print()
    
    # Generate comprehensive report
    analyzer.generate_shap_report()
    
    print("="*70)
    print("🎉 SHAP Analysis Complete!")
    print(f"📁 Results saved in: {analyzer.results_folder}")
    print("="*70)

if __name__ == "__main__":
    main()
