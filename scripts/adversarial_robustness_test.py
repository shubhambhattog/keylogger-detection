"""
Adversarial Robustness Testing for Keylogger Detection
======================================================
Test how well the enhanced model resists adversarial attacks
where attackers try to fool the model by slightly modifying features.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class AdversarialRobustnessTest:
    def __init__(self, dataset_folder="../dataset", results_folder="../results"):
        self.dataset_folder = dataset_folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)
        
    def load_enhanced_model_data(self, sample_size=1000):
        """Load enhanced data and train model"""
        print("📊 Loading enhanced dataset for robustness testing...")
        
        X_train = pd.read_csv(os.path.join(self.dataset_folder, "X_train_enhanced.csv"))
        X_test = pd.read_csv(os.path.join(self.dataset_folder, "X_test_enhanced.csv"))
        y_train = pd.read_csv(os.path.join(self.dataset_folder, "y_train_enhanced.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(self.dataset_folder, "y_test_enhanced.csv")).values.ravel()
        
        # Clean column names
        X_train.columns = X_train.columns.str.strip()
        X_test.columns = X_test.columns.str.strip()
        
        # Sample for robustness testing
        test_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_test_sample = X_test.iloc[test_indices]
        y_test_sample = y_test[test_indices]
        
        print(f"✅ Data loaded: {X_train.shape[0]:,} train, {len(X_test_sample):,} test samples")
        print(f"   Features: {X_train.shape[1]}")
        
        return X_train, X_test_sample, y_train, y_test_sample
    
    def train_robust_model(self, X_train, y_train):
        """Train the enhanced model for robustness testing"""
        print("\n🚀 Training enhanced model for robustness testing...")
        
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        print("✅ Enhanced model trained for robustness testing")
        return model
    
    def generate_adversarial_samples(self, X_test, epsilon_levels):
        """Generate adversarial samples with different noise levels"""
        print("\n🔍 Generating adversarial samples...")
        
        adversarial_samples = {}
        
        for epsilon in epsilon_levels:
            print(f"   Creating samples with ε={epsilon}")
            
            # Generate random noise
            noise = np.random.normal(0, epsilon, X_test.shape)
            
            # Add noise to original samples
            X_adversarial = X_test + noise
            
            # Store adversarial samples
            adversarial_samples[epsilon] = X_adversarial
        
        print(f"✅ Generated adversarial samples for {len(epsilon_levels)} noise levels")
        return adversarial_samples
    
    def test_robustness(self, model, X_test_clean, y_test, adversarial_samples):
        """Test model robustness against adversarial samples"""
        print("\n🛡️  Testing adversarial robustness...")
        
        results = {}
        
        # Test on clean data (baseline)
        clean_pred = model.predict(X_test_clean)
        results['clean'] = {
            'epsilon': 0.0,
            'accuracy': accuracy_score(y_test, clean_pred),
            'precision': precision_score(y_test, clean_pred, average='weighted'),
            'recall': recall_score(y_test, clean_pred, average='weighted'),
            'f1': f1_score(y_test, clean_pred, average='weighted')
        }
        
        print(f"   Clean accuracy: {results['clean']['accuracy']:.4f}")
        
        # Test on adversarial samples
        for epsilon, X_adversarial in adversarial_samples.items():
            try:
                adv_pred = model.predict(X_adversarial)
                
                results[f'epsilon_{epsilon}'] = {
                    'epsilon': epsilon,
                    'accuracy': accuracy_score(y_test, adv_pred),
                    'precision': precision_score(y_test, adv_pred, average='weighted'),
                    'recall': recall_score(y_test, adv_pred, average='weighted'),
                    'f1': f1_score(y_test, adv_pred, average='weighted')
                }
                
                print(f"   ε={epsilon}: accuracy={results[f'epsilon_{epsilon}']['accuracy']:.4f}")
                
            except Exception as e:
                print(f"   ⚠️ Error testing ε={epsilon}: {e}")
                continue
        
        print("✅ Robustness testing complete")
        return results
    
    def analyze_attack_success_rate(self, model, X_test_clean, y_test, adversarial_samples):
        """Analyze how successful adversarial attacks are"""
        print("\n📊 Analyzing attack success rates...")
        
        # Get clean predictions
        clean_pred = model.predict(X_test_clean)
        clean_correct = (clean_pred == y_test)
        
        attack_analysis = {}
        
        for epsilon, X_adversarial in adversarial_samples.items():
            adv_pred = model.predict(X_adversarial)
            
            # Calculate attack success: samples that were correct but now wrong
            attack_successful = clean_correct & (adv_pred != y_test)
            attack_success_rate = attack_successful.sum() / clean_correct.sum()
            
            # Calculate robustness: samples that remain correct
            remaining_correct = (adv_pred == y_test)
            robustness_rate = remaining_correct.sum() / len(y_test)
            
            attack_analysis[epsilon] = {
                'attack_success_rate': attack_success_rate,
                'robustness_rate': robustness_rate,
                'samples_flipped': attack_successful.sum(),
                'total_correct_originally': clean_correct.sum()
            }
            
            print(f"   ε={epsilon}: {attack_success_rate:.2%} attack success, {robustness_rate:.2%} robustness")
        
        return attack_analysis
    
    def create_robustness_visualizations(self, results, attack_analysis):
        """Create visualizations for robustness analysis"""
        print("\n📊 Creating robustness visualizations...")
        
        # Prepare data
        epsilon_values = [results[key]['epsilon'] for key in results.keys()]
        accuracy_values = [results[key]['accuracy'] for key in results.keys()]
        precision_values = [results[key]['precision'] for key in results.keys()]
        recall_values = [results[key]['recall'] for key in results.keys()]
        f1_values = [results[key]['f1'] for key in results.keys()]
        
        # 1. Robustness curve
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy degradation
        axes[0,0].plot(epsilon_values, accuracy_values, 'b-o', linewidth=2, markersize=6)
        axes[0,0].set_xlabel('Epsilon (Noise Level)')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Accuracy vs Adversarial Noise', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim([0, 1])
        
        # F1-Score degradation
        axes[0,1].plot(epsilon_values, f1_values, 'r-s', linewidth=2, markersize=6)
        axes[0,1].set_xlabel('Epsilon (Noise Level)')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_title('F1-Score vs Adversarial Noise', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim([0, 1])
        
        # Attack success rate
        attack_epsilons = list(attack_analysis.keys())
        attack_success_rates = [attack_analysis[eps]['attack_success_rate'] for eps in attack_epsilons]
        
        axes[1,0].plot(attack_epsilons, attack_success_rates, 'g-^', linewidth=2, markersize=6)
        axes[1,0].set_xlabel('Epsilon (Noise Level)')
        axes[1,0].set_ylabel('Attack Success Rate')
        axes[1,0].set_title('Adversarial Attack Success Rate', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim([0, 1])
        
        # Robustness comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        clean_scores = [results['clean']['accuracy'], results['clean']['precision'], 
                       results['clean']['recall'], results['clean']['f1']]
        worst_case_key = max(results.keys(), key=lambda x: results[x]['epsilon'])
        worst_scores = [results[worst_case_key]['accuracy'], results[worst_case_key]['precision'],
                       results[worst_case_key]['recall'], results[worst_case_key]['f1']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1,1].bar(x - width/2, clean_scores, width, label='Clean Data', alpha=0.8, color='lightblue')
        axes[1,1].bar(x + width/2, worst_scores, width, label=f'Adversarial (ε={results[worst_case_key]["epsilon"]})', 
                     alpha=0.8, color='lightcoral')
        
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Clean vs Adversarial Performance', fontweight='bold')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(metrics)
        axes[1,1].legend()
        axes[1,1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/adversarial_robustness_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Saved: adversarial_robustness_analysis.png")
        
        # 2. Robustness summary table visualization
        plt.figure(figsize=(12, 8))
        
        # Create heatmap data
        heatmap_data = []
        for key in results.keys():
            if key != 'clean':
                epsilon = results[key]['epsilon']
                accuracy = results[key]['accuracy']
                f1 = results[key]['f1']
                attack_success = attack_analysis.get(epsilon, {}).get('attack_success_rate', 0)
                
                heatmap_data.append([epsilon, accuracy, f1, 1-attack_success])  # Robustness = 1 - attack success
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, columns=['Epsilon', 'Accuracy', 'F1-Score', 'Robustness'])
            
            # Create heatmap
            sns.heatmap(heatmap_df.set_index('Epsilon').T, annot=True, cmap='RdYlGn', 
                       vmin=0, vmax=1, fmt='.3f', cbar_kws={'label': 'Performance Score'})
            plt.title('Adversarial Robustness Heatmap\n(Green = Robust, Red = Vulnerable)', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('Metrics')
            plt.xlabel('Noise Level (Epsilon)')
            
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/robustness_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("   ✅ Saved: robustness_heatmap.png")
    
    def generate_robustness_report(self, results, attack_analysis):
        """Generate comprehensive robustness report"""
        print("\n📄 Generating robustness analysis report...")
        
        # Find critical points
        clean_accuracy = results['clean']['accuracy']
        worst_epsilon = max([results[key]['epsilon'] for key in results.keys()])
        worst_accuracy = min([results[key]['accuracy'] for key in results.keys()])
        
        robustness_threshold = 0.05  # 5% degradation threshold
        critical_epsilon = None
        for key in sorted(results.keys(), key=lambda x: results[x]['epsilon']):
            if results[key]['accuracy'] < clean_accuracy - robustness_threshold:
                critical_epsilon = results[key]['epsilon']
                break
        
        report_content = f"""
# Adversarial Robustness Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Security Assessment

### Baseline Performance (Clean Data)
- **Accuracy**: {clean_accuracy:.4f}
- **Precision**: {results['clean']['precision']:.4f}
- **Recall**: {results['clean']['recall']:.4f}
- **F1-Score**: {results['clean']['f1']:.4f}

### Robustness Analysis Results

#### Overall Robustness
- **Maximum noise tested**: ε = {worst_epsilon}
- **Worst-case accuracy**: {worst_accuracy:.4f}
- **Performance degradation**: {((clean_accuracy - worst_accuracy) / clean_accuracy * 100):.2f}%
- **Critical noise level**: ε = {critical_epsilon if critical_epsilon else 'Above tested range'} (5% degradation threshold)

#### Attack Success Analysis
"""
        
        for epsilon in sorted(attack_analysis.keys()):
            stats = attack_analysis[epsilon]
            report_content += f"""
**Noise Level ε = {epsilon}**
- Attack success rate: {stats['attack_success_rate']:.2%}
- Samples successfully attacked: {stats['samples_flipped']}/{stats['total_correct_originally']}
- Model robustness: {stats['robustness_rate']:.2%}
"""
        
        report_content += """
## Security Implications

### Robustness Assessment
"""
        
        if worst_accuracy > 0.95:
            assessment = "EXCELLENT"
            description = "Model shows exceptional resistance to adversarial attacks"
        elif worst_accuracy > 0.90:
            assessment = "GOOD"
            description = "Model demonstrates good robustness against adversarial perturbations"
        elif worst_accuracy > 0.80:
            assessment = "MODERATE"
            description = "Model shows moderate vulnerability to adversarial attacks"
        else:
            assessment = "POOR"
            description = "Model is highly vulnerable to adversarial attacks"
        
        report_content += f"""
**Overall Assessment**: {assessment}
{description}

### Key Findings
1. **Noise Tolerance**: Model maintains performance up to ε = {critical_epsilon if critical_epsilon else 'high levels'}
2. **Attack Resistance**: {"Strong" if worst_accuracy > 0.90 else "Moderate" if worst_accuracy > 0.80 else "Weak"} resistance to adversarial manipulation
3. **Critical Vulnerabilities**: {"None identified" if critical_epsilon is None else f"Performance degrades significantly at ε = {critical_epsilon}"}

### Recommendations
1. **Deployment Readiness**: {"Ready for production deployment" if worst_accuracy > 0.90 else "Consider additional hardening"}
2. **Monitoring**: Implement input validation and anomaly detection
3. **Defense Strategy**: {"Current robustness is adequate" if worst_accuracy > 0.90 else "Consider adversarial training"}

## Technical Details
- **Testing Method**: Gaussian noise injection
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Attack Model**: Untargeted adversarial perturbations
- **Sample Size**: {len(attack_analysis[list(attack_analysis.keys())[0]]) if attack_analysis else 'N/A'}
"""
        
        # Save report
        with open(f'{self.results_folder}/adversarial_robustness_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed results
        results_df = pd.DataFrame([
            {
                'epsilon': results[key]['epsilon'],
                'accuracy': results[key]['accuracy'],
                'precision': results[key]['precision'],
                'recall': results[key]['recall'],
                'f1_score': results[key]['f1'],
                'attack_success_rate': attack_analysis.get(results[key]['epsilon'], {}).get('attack_success_rate', 0)
            }
            for key in results.keys()
        ])
        
        results_df.to_csv(f'{self.results_folder}/robustness_test_results.csv', index=False)
        
        print("   ✅ Saved: adversarial_robustness_report.md")
        print("   ✅ Saved: robustness_test_results.csv")
        
        return assessment, worst_accuracy
    
    def run_adversarial_testing(self):
        """Run complete adversarial robustness testing"""
        print("🛡️  Adversarial Robustness Testing - Enhanced Keylogger Detection Model")
        print("="*80)
        
        # Load data and train model
        X_train, X_test, y_train, y_test = self.load_enhanced_model_data()
        model = self.train_robust_model(X_train, y_train)
        
        # Define noise levels to test
        epsilon_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        print(f"\n📊 Testing noise levels: {epsilon_levels}")
        
        # Generate adversarial samples
        adversarial_samples = self.generate_adversarial_samples(X_test, epsilon_levels)
        
        # Test robustness
        results = self.test_robustness(model, X_test, y_test, adversarial_samples)
        
        # Analyze attack success
        attack_analysis = self.analyze_attack_success_rate(model, X_test, y_test, adversarial_samples)
        
        # Create visualizations
        self.create_robustness_visualizations(results, attack_analysis)
        
        # Generate report
        assessment, worst_accuracy = self.generate_robustness_report(results, attack_analysis)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 Adversarial Robustness Testing Complete!")
        print(f"🛡️  Overall Assessment: {assessment}")
        print(f"📊 Clean Accuracy: {results['clean']['accuracy']:.4f}")
        print(f"📊 Worst-case Accuracy: {worst_accuracy:.4f}")
        print(f"📉 Performance Degradation: {((results['clean']['accuracy'] - worst_accuracy) / results['clean']['accuracy'] * 100):.2f}%")
        
        print("\n📁 Files Generated:")
        print("   • adversarial_robustness_analysis.png - Performance curves")
        print("   • robustness_heatmap.png - Security assessment heatmap") 
        print("   • adversarial_robustness_report.md - Comprehensive security report")
        print("   • robustness_test_results.csv - Detailed test results")
        print("="*80)
        
        return assessment, results

def main():
    tester = AdversarialRobustnessTest()
    tester.run_adversarial_testing()

if __name__ == "__main__":
    main()
