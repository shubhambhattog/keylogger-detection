"""
Advanced Feature Engineering for Keylogger Detection
==================================================

This module creates sophisticated features based on SHAP analysis insights.
SHAP showed that seq, sbytes, dur, and rate are most important - so we'll 
create advanced features that capture these patterns more effectively.

Key Focus Areas (from SHAP):
1. Sequence patterns (seq: 0.824 importance)
2. Source bytes patterns (sbytes: 0.477 importance) 
3. Duration patterns (dur: 0.287 importance)
4. Rate patterns (rate: 0.263 importance)

Author: Your Name
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering based on SHAP analysis insights
    """
    
    def __init__(self, dataset_folder="../dataset"):
        self.dataset_folder = dataset_folder
        self.original_features = []
        self.new_features = []
        
    def load_data(self):
        """Load the original dataset"""
        print("🔄 Loading original dataset...")
        
        # Load training and test data
        self.X_train = pd.read_csv(os.path.join(self.dataset_folder, "X_train.csv"))
        self.X_test = pd.read_csv(os.path.join(self.dataset_folder, "X_test.csv"))
        self.y_train = pd.read_csv(os.path.join(self.dataset_folder, "y_train.csv")).values.ravel()
        self.y_test = pd.read_csv(os.path.join(self.dataset_folder, "y_test.csv")).values.ravel()
        
        # Clean column names
        self.X_train.columns = self.X_train.columns.str.strip()
        self.X_test.columns = self.X_test.columns.str.strip()
        
        self.original_features = list(self.X_train.columns)
        
        print(f"✅ Dataset loaded!")
        print(f"   Training samples: {len(self.X_train):,}")
        print(f"   Test samples: {len(self.X_test):,}")
        print(f"   Original features: {len(self.original_features)}")
        
    def create_sequence_features(self, df):
        """
        Create advanced sequence-based features
        SHAP showed 'seq' is most important (0.824)
        """
        print("🔄 Creating sequence-based features...")
        
        # 1. Sequence regularity score
        df['seq_regularity'] = np.where(df['seq'] != 0, 1.0 / (1.0 + np.abs(df['seq'])), 0)
        
        # 2. Sequence anomaly detection (how far from expected sequence)
        df['seq_anomaly_score'] = np.abs(df['seq'] - df['seq'].median()) / (df['seq'].std() + 1e-6)
        
        # 3. Sequence entropy (randomness)
        # Bin sequences into groups and calculate entropy
        seq_binned = pd.cut(df['seq'], bins=10, labels=False)
        df['seq_entropy'] = seq_binned.apply(lambda x: entropy(np.bincount(seq_binned + 1)))
        
        # 4. Sequence momentum (trend detection)
        df['seq_momentum'] = df['seq'].rolling(window=5, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0
        )
        
        # 5. Sequence volatility
        df['seq_volatility'] = df['seq'].rolling(window=10, min_periods=1).std()
        df['seq_volatility'] = df['seq_volatility'].fillna(0)
        
        print(f"   ✅ Created 5 sequence-based features")
        
    def create_bytes_features(self, df):
        """
        Create advanced byte-pattern features
        SHAP showed 'sbytes' is very important (0.477)
        """
        print("🔄 Creating byte-pattern features...")
        
        # 1. Byte transmission regularity
        df['byte_regularity_score'] = 1.0 / (1.0 + df['sbytes'].std() / (df['sbytes'].mean() + 1e-6))
        
        # 2. Small packet indicator (keyloggers send small, regular packets)
        small_packet_threshold = df['sbytes'].quantile(0.25)
        df['small_packet_ratio'] = (df['sbytes'] <= small_packet_threshold).astype(int)
        
        # 3. Byte pattern entropy
        # Create histogram of byte sizes and calculate entropy
        byte_bins = pd.cut(df['sbytes'], bins=20, labels=False)
        df['sbytes_entropy'] = byte_bins.apply(lambda x: entropy(np.bincount(byte_bins + 1)))
        
        # 4. Source vs destination byte asymmetry
        df['byte_asymmetry'] = (df['sbytes'] - df['dbytes']) / (df['sbytes'] + df['dbytes'] + 1e-6)
        
        # 5. Byte efficiency score (bytes per packet)
        df['byte_efficiency'] = df['sbytes'] / (df['spkts'] + 1e-6)
        
        # 6. Consistent byte size indicator
        df['byte_consistency'] = np.exp(-df['sbytes'].std() / (df['sbytes'].mean() + 1e-6))
        
        print(f"   ✅ Created 6 byte-pattern features")
        
    def create_timing_features(self, df):
        """
        Create advanced timing-based features
        SHAP showed 'dur' and 'rate' are important (0.287, 0.263)
        """
        print("🔄 Creating timing-based features...")
        
        # 1. Persistent connection indicator
        long_duration_threshold = df['dur'].quantile(0.75)
        df['persistent_connection'] = (df['dur'] >= long_duration_threshold).astype(int)
        
        # 2. Communication rhythm score
        # Keyloggers have steady, predictable timing
        df['rhythm_score'] = 1.0 / (1.0 + df['rate'].std() / (df['rate'].mean() + 1e-6))
        
        # 3. Burst vs steady transmission
        rate_median = df['rate'].median()
        df['burst_indicator'] = (df['rate'] > 2 * rate_median).astype(int)
        df['steady_transmission'] = 1 - df['burst_indicator']
        
        # 4. Timing predictability
        df['timing_predictability'] = np.exp(-df['dur'].std() / (df['dur'].mean() + 1e-6))
        
        # 5. Rate consistency
        df['rate_consistency'] = 1.0 / (1.0 + np.abs(df['rate'] - df['rate'].median()) / (df['rate'].std() + 1e-6))
        
        # 6. Duration anomaly
        df['duration_anomaly'] = np.abs(df['dur'] - df['dur'].median()) / (df['dur'].std() + 1e-6)
        
        # 7. Suspicious timing pattern (very long + very slow = keylogger-like)
        df['suspicious_timing'] = ((df['dur'] > df['dur'].quantile(0.8)) & 
                                  (df['rate'] < df['rate'].quantile(0.2))).astype(int)
        
        print(f"   ✅ Created 7 timing-based features")
        
    def create_statistical_features(self, df):
        """
        Create advanced statistical features
        SHAP showed 'sum', 'min', 'max' are moderately important
        """
        print("🔄 Creating statistical features...")
        
        # 1. Statistical stability score
        df['stat_stability'] = 1.0 / (1.0 + df['stddev'] / (df['mean'] + 1e-6))
        
        # 2. Range normalization
        df['normalized_range'] = (df['max'] - df['min']) / (df['sum'] + 1e-6)
        
        # 3. Coefficient of variation
        df['coeff_variation'] = df['stddev'] / (df['mean'] + 1e-6)
        
        # 4. Skewness indicator
        # Calculate skewness from available statistics (approximation)
        df['stat_skewness'] = (df['mean'] - df['min']) / (df['max'] - df['min'] + 1e-6)
        
        # 5. Statistical consistency
        expected_mean = (df['min'] + df['max']) / 2
        df['mean_consistency'] = 1.0 / (1.0 + np.abs(df['mean'] - expected_mean) / (df['stddev'] + 1e-6))
        
        print(f"   ✅ Created 5 statistical features")
        
    def create_behavioral_features(self, df):
        """
        Create features that capture keylogger behavioral patterns
        """
        print("🔄 Creating behavioral pattern features...")
        
        # 1. Keylogger signature score (combination of suspicious patterns)
        df['keylogger_signature'] = (
            df['small_packet_ratio'] * 0.3 +
            df['persistent_connection'] * 0.25 +
            df['steady_transmission'] * 0.2 +
            df['rhythm_score'] * 0.15 +
            df['stat_stability'] * 0.1
        )
        
        # 2. Stealth communication indicator
        df['stealth_communication'] = (
            (df['rate'] < df['rate'].quantile(0.3)) &  # Slow transmission
            (df['dur'] > df['dur'].quantile(0.7)) &   # Long duration
            (df['sbytes'] < df['sbytes'].quantile(0.4))  # Small packets
        ).astype(int)
        
        # 3. Communication efficiency
        df['communication_efficiency'] = (df['bytes'] / (df['dur'] + 1e-6)) / (df['pkts'] + 1e-6)
        
        # 4. Pattern regularity index
        df['pattern_regularity'] = (
            df['seq_regularity'] * 0.4 +
            df['byte_consistency'] * 0.35 +
            df['timing_predictability'] * 0.25
        )
        
        print(f"   ✅ Created 4 behavioral features")
        
    def create_all_features(self, df):
        """Create all advanced features for a dataset"""
        df_enhanced = df.copy()
        
        # Create each category of features
        self.create_sequence_features(df_enhanced)
        self.create_bytes_features(df_enhanced)
        self.create_timing_features(df_enhanced)
        self.create_statistical_features(df_enhanced)
        self.create_behavioral_features(df_enhanced)
        
        # Handle any infinite or NaN values
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
        df_enhanced = df_enhanced.fillna(0)
        
        return df_enhanced
        
    def engineer_features(self):
        """Main feature engineering pipeline"""
        print("🚀 Starting Advanced Feature Engineering")
        print("="*60)
        print("Based on SHAP insights: seq(0.824), sbytes(0.477), dur(0.287), rate(0.263)")
        print()
        
        # Create enhanced datasets
        print("📊 Enhancing training dataset...")
        self.X_train_enhanced = self.create_all_features(self.X_train)
        
        print("\n📊 Enhancing test dataset...")
        self.X_test_enhanced = self.create_all_features(self.X_test)
        
        # Get list of new features
        self.new_features = [col for col in self.X_train_enhanced.columns 
                           if col not in self.original_features]
        
        print(f"\n✅ Feature Engineering Complete!")
        print(f"   Original features: {len(self.original_features)}")
        print(f"   New features created: {len(self.new_features)}")
        print(f"   Total features: {len(self.X_train_enhanced.columns)}")
        
        # Display new features
        print(f"\n🆕 New Features Created:")
        for i, feature in enumerate(self.new_features, 1):
            print(f"   {i:2d}. {feature}")
        
        return self.X_train_enhanced, self.X_test_enhanced
        
    def save_enhanced_data(self, output_folder="../dataset"):
        """Save the enhanced datasets"""
        print(f"\n💾 Saving enhanced datasets to {output_folder}...")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Save enhanced datasets
        self.X_train_enhanced.to_csv(os.path.join(output_folder, "X_train_enhanced.csv"), index=False)
        self.X_test_enhanced.to_csv(os.path.join(output_folder, "X_test_enhanced.csv"), index=False)
        
        # Save feature lists
        feature_info = pd.DataFrame({
            'Feature_Name': self.X_train_enhanced.columns,
            'Feature_Type': ['Original' if f in self.original_features else 'Engineered' 
                           for f in self.X_train_enhanced.columns]
        })
        feature_info.to_csv(os.path.join(output_folder, "feature_info.csv"), index=False)
        
        print(f"   ✅ Saved X_train_enhanced.csv ({self.X_train_enhanced.shape})")
        print(f"   ✅ Saved X_test_enhanced.csv ({self.X_test_enhanced.shape})")
        print(f"   ✅ Saved feature_info.csv")
        
    def create_feature_analysis_report(self, output_folder="../results"):
        """Create a report analyzing the new features"""
        print(f"\n📊 Creating feature analysis report...")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Analyze new feature distributions
        new_features_df = self.X_train_enhanced[self.new_features]
        
        # Create summary statistics
        feature_stats = new_features_df.describe().T
        feature_stats['missing_values'] = new_features_df.isnull().sum()
        feature_stats['unique_values'] = new_features_df.nunique()
        
        # Save statistics
        feature_stats.to_csv(os.path.join(output_folder, "new_features_statistics.csv"))
        
        # Create correlation analysis
        feature_correlations = new_features_df.corr()
        
        # Visualize correlations
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(feature_correlations))
        sns.heatmap(feature_correlations, mask=mask, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of New Engineered Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "new_features_correlation.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved new_features_statistics.csv")
        print(f"   ✅ Saved new_features_correlation.png")
        
        # Create feature distribution plots for key features
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        key_features = [
            'keylogger_signature', 'stealth_communication', 'pattern_regularity',
            'seq_regularity', 'byte_consistency', 'rhythm_score',
            'suspicious_timing', 'stat_stability', 'communication_efficiency'
        ]
        
        for i, feature in enumerate(key_features[:9]):
            if feature in new_features_df.columns:
                axes[i].hist(new_features_df[feature], bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature}', fontweight='bold')
                axes[i].set_xlabel('Feature Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Distribution of Key Engineered Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "new_features_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved new_features_distributions.png")
        
def main():
    """Main execution function"""
    print("🎯 Advanced Feature Engineering for Keylogger Detection")
    print("="*70)
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineering()
    
    # Load data
    feature_engineer.load_data()
    
    # Engineer features
    X_train_enhanced, X_test_enhanced = feature_engineer.engineer_features()
    
    # Save results
    feature_engineer.save_enhanced_data()
    feature_engineer.create_feature_analysis_report()
    
    print("\n" + "="*70)
    print("🎉 Advanced Feature Engineering Complete!")
    print("📁 Next steps: Train models with enhanced features and compare performance")
    print("="*70)

if __name__ == "__main__":
    main()
