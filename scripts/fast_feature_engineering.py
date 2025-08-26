"""
Fast Advanced Feature Engineering for Keylogger Detection
========================================================
Optimized version that processes data efficiently using vectorization
and sampling for quick results.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

class FastFeatureEngineer:
    def __init__(self, dataset_folder="../dataset"):
        self.dataset_folder = dataset_folder
        
    def load_sample_data(self, sample_size=50000):
        """Load a sample of data for faster processing"""
        print(f"🔄 Loading sample data ({sample_size:,} samples)...")
        
        # Load original data
        X_train = pd.read_csv(os.path.join(self.dataset_folder, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(self.dataset_folder, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(self.dataset_folder, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(self.dataset_folder, "y_test.csv")).values.ravel()
        
        # Sample for faster processing
        train_sample_size = min(sample_size // 2, len(X_train))
        test_sample_size = min(sample_size // 2, len(X_test))
        
        train_indices = np.random.choice(len(X_train), train_sample_size, replace=False)
        test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
        
        X_train_sample = X_train.iloc[train_indices].copy()
        X_test_sample = X_test.iloc[test_indices].copy()
        y_train_sample = y_train[train_indices]
        y_test_sample = y_test[test_indices]
        
        print(f"✅ Loaded sample data: Train {len(X_train_sample):,}, Test {len(X_test_sample):,}")
        
        return X_train_sample, X_test_sample, y_train_sample, y_test_sample
    
    def create_timing_features(self, df):
        """Create advanced timing-based features (vectorized)"""
        print("🔄 Creating timing features...")
        start_time = time.time()
        
        # Keystroke rhythm indicators (based on SHAP insights)
        df['flow_regularity'] = np.where(df['dur'] > 0, df['pkts'] / (df['dur'] + 1e-8), 0)
        df['burst_intensity'] = np.where(df['dur'] > 0, df['bytes'] / (df['dur'] + 1e-8), 0)
        df['packet_rhythm'] = np.where(df['pkts'] > 0, df['dur'] / (df['pkts'] + 1e-8), 0)
        
        # Communication pattern detection
        df['sustained_connection'] = (df['dur'] > df['dur'].quantile(0.8)).astype(int)
        df['rapid_transmission'] = (df['rate'] > df['rate'].quantile(0.9)).astype(int)
        
        print(f"   ✅ Timing features created in {time.time() - start_time:.2f}s")
        return df
    
    def create_entropy_features(self, df):
        """Create entropy and randomness features (optimized)"""
        print("🔄 Creating entropy features...")
        start_time = time.time()
        
        # Simplified entropy approximations (much faster than full entropy)
        df['size_variation'] = df['stddev'] / (df['mean'] + 1e-8)
        df['flow_predictability'] = df['min'] / (df['max'] + 1e-8)
        df['pattern_consistency'] = df['mean'] / (df['sum'] + 1e-8)
        
        # Traffic randomness indicators
        df['directional_bias'] = np.abs(df['spkts'] - df['dpkts']) / (df['pkts'] + 1e-8)
        df['byte_efficiency'] = df['bytes'] / (df['pkts'] + 1e-8)
        
        print(f"   ✅ Entropy features created in {time.time() - start_time:.2f}s")
        return df
    
    def create_behavioral_features(self, df):
        """Create keylogger behavior detection features"""
        print("🔄 Creating behavioral features...")
        start_time = time.time()
        
        # Keylogger-specific patterns (based on SHAP seq, sbytes importance)
        df['small_packet_ratio'] = (df['sbytes'] < df['sbytes'].quantile(0.3)).astype(int)
        df['persistent_flow'] = ((df['dur'] > df['dur'].median()) & 
                                (df['rate'] < df['rate'].median())).astype(int)
        
        # Communication stealth indicators
        df['stealth_score'] = (df['bytes'] / (df['pkts'] + 1e-8)) * (df['dur'] + 1e-8) / (df['rate'] + 1e-8)
        df['keylogger_signature'] = ((df['sbytes'] > 0) & (df['sbytes'] < df['sbytes'].quantile(0.5)) & 
                                    (df['dur'] > df['dur'].quantile(0.6))).astype(int)
        
        # Advanced statistical patterns
        df['flow_skewness'] = df['max'] - df['min'] / (df['stddev'] + 1e-8)
        df['transmission_efficiency'] = df['bytes'] / (df['dur'] + 1e-8)
        
        print(f"   ✅ Behavioral features created in {time.time() - start_time:.2f}s")
        return df
    
    def create_interaction_features(self, df):
        """Create feature interactions (top SHAP features)"""
        print("🔄 Creating interaction features...")
        start_time = time.time()
        
        # Based on top SHAP features: seq, sbytes, dur, rate, sum
        df['seq_sbytes_ratio'] = df['seq'] * df['sbytes']
        df['dur_rate_product'] = df['dur'] * df['rate'] 
        df['sbytes_dur_ratio'] = df['sbytes'] / (df['dur'] + 1e-8)
        df['sum_seq_interaction'] = df['sum'] * df['seq']
        
        # Advanced combinations
        df['keylogger_composite'] = (df['seq'] * 0.8 + df['sbytes'] * 0.4 + 
                                    df['dur'] * 0.3 + df['rate'] * 0.2) / 4
        
        print(f"   ✅ Interaction features created in {time.time() - start_time:.2f}s")
        return df
    
    def process_dataset(self, df):
        """Apply all feature engineering steps"""
        print(f"🔧 Processing dataset with {df.shape[0]:,} samples, {df.shape[1]} features...")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Create new features
        df = self.create_timing_features(df)
        df = self.create_entropy_features(df)  
        df = self.create_behavioral_features(df)
        df = self.create_interaction_features(df)
        
        # Handle any infinities or NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        print(f"✅ Feature engineering complete: {df.shape[0]:,} samples, {df.shape[1]} features")
        return df
    
    def run_fast_feature_engineering(self, sample_size=50000):
        """Run the complete fast feature engineering pipeline"""
        print("🚀 Fast Advanced Feature Engineering Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load sample data
        X_train, X_test, y_train, y_test = self.load_sample_data(sample_size)
        
        # Process datasets
        print("\n📊 Processing training data...")
        X_train_enhanced = self.process_dataset(X_train)
        
        print("\n📊 Processing test data...")
        X_test_enhanced = self.process_dataset(X_test)
        
        # Save enhanced datasets
        print("\n💾 Saving enhanced datasets...")
        X_train_enhanced.to_csv(f"{self.dataset_folder}/X_train_enhanced.csv", index=False)
        X_test_enhanced.to_csv(f"{self.dataset_folder}/X_test_enhanced.csv", index=False)
        
        pd.DataFrame(y_train, columns=['target']).to_csv(f"{self.dataset_folder}/y_train_enhanced.csv", index=False)
        pd.DataFrame(y_test, columns=['target']).to_csv(f"{self.dataset_folder}/y_test_enhanced.csv", index=False)
        
        total_time = time.time() - start_time
        
        # Summary report
        print("\n" + "=" * 60)
        print("🎉 Feature Engineering Complete!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📊 Original features: {X_train.shape[1]}")
        print(f"🆕 Enhanced features: {X_train_enhanced.shape[1]}")
        print(f"➕ New features added: {X_train_enhanced.shape[1] - X_train.shape[1]}")
        print(f"📁 Files saved:")
        print(f"   • X_train_enhanced.csv ({X_train_enhanced.shape[0]:,} samples)")
        print(f"   • X_test_enhanced.csv ({X_test_enhanced.shape[0]:,} samples)")
        print(f"   • y_train_enhanced.csv & y_test_enhanced.csv")
        
        # Feature summary
        new_features = [col for col in X_train_enhanced.columns if col not in X_train.columns]
        print(f"\n🆕 New features created ({len(new_features)}):")
        for i, feature in enumerate(new_features, 1):
            print(f"   {i:2d}. {feature}")
        
        return X_train_enhanced, X_test_enhanced, y_train, y_test

def main():
    engineer = FastFeatureEngineer()
    X_train_enhanced, X_test_enhanced, y_train, y_test = engineer.run_fast_feature_engineering(sample_size=100000)

if __name__ == "__main__":
    main()
