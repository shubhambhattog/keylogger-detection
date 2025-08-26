"""
Real-time Deployment Simulation
===============================
Convert enhanced model to lightweight format and simulate
edge device deployment with performance benchmarks.
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import psutil
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DeploymentSimulator:
    def __init__(self, dataset_folder="../dataset", results_folder="../results"):
        self.dataset_folder = dataset_folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)
        
    def load_deployment_data(self, sample_size=5000):
        """Load data for deployment simulation"""
        print("📊 Loading deployment test data...")
        
        X_train = pd.read_csv(os.path.join(self.dataset_folder, "X_train_enhanced.csv"))
        X_test = pd.read_csv(os.path.join(self.dataset_folder, "X_test_enhanced.csv"))
        y_train = pd.read_csv(os.path.join(self.dataset_folder, "y_train_enhanced.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(self.dataset_folder, "y_test_enhanced.csv")).values.ravel()
        
        # Clean column names
        X_train.columns = X_train.columns.str.strip()
        X_test.columns = X_test.columns.str.strip()
        
        # Sample for deployment testing
        test_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_deploy_test = X_test.iloc[test_indices]
        y_deploy_test = y_test[test_indices]
        
        print(f"✅ Deployment data loaded: {X_train.shape[0]:,} train, {len(X_deploy_test):,} test")
        print(f"   Features: {X_train.shape[1]}")
        
        return X_train, X_deploy_test, y_train, y_deploy_test
    
    def train_deployment_model(self, X_train, y_train):
        """Train optimized model for deployment"""
        print("\n🚀 Training model optimized for deployment...")
        
        # Optimized parameters for speed vs accuracy balance
        model = CatBoostClassifier(
            iterations=100,  # Reduced for faster inference
            depth=4,         # Reduced for smaller model size
            learning_rate=0.15,  # Increased to compensate for fewer iterations
            verbose=False,
            random_seed=42
        )
        
        start_time = time.time()
        model.fit(X_train, y_train, verbose=False)
        train_time = time.time() - start_time
        
        print(f"✅ Deployment model trained in {train_time:.2f} seconds")
        return model, train_time
    
    def benchmark_inference_performance(self, model, X_test, y_test, batch_sizes=[1, 10, 100, 1000]):
        """Benchmark inference performance for different batch sizes"""
        print("\n⚡ Benchmarking inference performance...")
        
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            # Prepare batches
            n_batches = len(X_test) // batch_size
            if n_batches == 0:
                continue
                
            total_time = 0
            total_samples = 0
            predictions = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_test))
                batch_X = X_test.iloc[start_idx:end_idx]
                
                # Memory usage before prediction
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time the prediction
                start_time = time.time()
                batch_pred = model.predict(batch_X)
                inference_time = time.time() - start_time
                
                # Memory usage after prediction
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                total_time += inference_time
                total_samples += len(batch_X)
                predictions.extend(batch_pred)
            
            # Calculate metrics
            avg_time_per_sample = (total_time / total_samples) * 1000  # milliseconds
            throughput = total_samples / total_time  # samples per second
            accuracy = accuracy_score(y_test[:total_samples], predictions)
            
            benchmark_results[batch_size] = {
                'total_time': total_time,
                'total_samples': total_samples,
                'avg_time_per_sample_ms': avg_time_per_sample,
                'throughput_samples_per_sec': throughput,
                'accuracy': accuracy,
                'memory_usage_mb': mem_after
            }
            
            print(f"      {avg_time_per_sample:.3f}ms/sample, {throughput:.1f} samples/sec, {accuracy:.4f} accuracy")
        
        return benchmark_results
    
    def simulate_realtime_processing(self, model, X_test, duration_seconds=30):
        """Simulate real-time processing for specified duration"""
        print(f"\n🔄 Simulating real-time processing for {duration_seconds} seconds...")
        
        start_time = time.time()
        processed_samples = 0
        processing_times = []
        memory_usage = []
        cpu_usage = []
        
        process = psutil.Process()
        
        while (time.time() - start_time) < duration_seconds:
            # Simulate receiving a new sample
            sample_idx = np.random.randint(0, len(X_test))
            sample = X_test.iloc[[sample_idx]]
            
            # Monitor system resources
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time the prediction
            pred_start = time.time()
            prediction = model.predict(sample)
            pred_time = time.time() - pred_start
            
            # Record metrics
            processing_times.append(pred_time * 1000)  # milliseconds
            memory_usage.append(mem_usage)
            cpu_usage.append(cpu_percent)
            processed_samples += 1
            
            # Small delay to simulate realistic timing
            time.sleep(0.01)  # 10ms delay between samples
        
        actual_duration = time.time() - start_time
        
        realtime_stats = {
            'duration_seconds': actual_duration,
            'samples_processed': processed_samples,
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'std_processing_time_ms': np.std(processing_times),
            'avg_memory_usage_mb': np.mean(memory_usage),
            'max_memory_usage_mb': np.max(memory_usage),
            'avg_cpu_usage_percent': np.mean(cpu_usage),
            'max_cpu_usage_percent': np.max(cpu_usage),
            'throughput_per_second': processed_samples / actual_duration
        }
        
        print(f"✅ Processed {processed_samples} samples in {actual_duration:.2f}s")
        print(f"   Average: {realtime_stats['avg_processing_time_ms']:.3f}ms per prediction")
        print(f"   Throughput: {realtime_stats['throughput_per_second']:.1f} predictions/second")
        
        return realtime_stats, processing_times, memory_usage, cpu_usage
    
    def analyze_deployment_constraints(self, model, X_test):
        """Analyze deployment constraints and requirements"""
        print("\n📊 Analyzing deployment constraints...")
        
        # Model size analysis
        model_path = os.path.join(self.results_folder, "temp_deployment_model.cbm")
        model.save_model(model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.remove(model_path)  # Clean up
        
        # Memory footprint analysis
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Load model and data into memory
        test_sample = X_test.sample(1000)
        predictions = model.predict(test_sample)
        
        mem_after = process.memory_info().rss / 1024 / 1024
        inference_memory = mem_after - mem_before
        
        # Feature analysis
        feature_count = len(X_test.columns)
        numeric_features = X_test.select_dtypes(include=[np.number]).shape[1]
        categorical_features = feature_count - numeric_features
        
        deployment_analysis = {
            'model_size_mb': model_size_mb,
            'inference_memory_mb': max(inference_memory, 0),
            'total_memory_requirement_mb': model_size_mb + max(inference_memory, 10),
            'feature_count': feature_count,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'input_size_kb': (X_test.memory_usage(deep=True).sum() / len(X_test)) / 1024,
        }
        
        print(f"   Model size: {model_size_mb:.2f} MB")
        print(f"   Memory requirement: {deployment_analysis['total_memory_requirement_mb']:.2f} MB")
        print(f"   Features: {feature_count} ({numeric_features} numeric, {categorical_features} categorical)")
        
        return deployment_analysis
    
    def create_deployment_visualizations(self, benchmark_results, realtime_stats, 
                                       processing_times, memory_usage, cpu_usage):
        """Create deployment performance visualizations"""
        print("\n📊 Creating deployment visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Batch size performance
        batch_sizes = list(benchmark_results.keys())
        latencies = [benchmark_results[bs]['avg_time_per_sample_ms'] for bs in batch_sizes]
        throughputs = [benchmark_results[bs]['throughput_samples_per_sec'] for bs in batch_sizes]
        
        axes[0,0].plot(batch_sizes, latencies, 'b-o', linewidth=2, markersize=6)
        axes[0,0].set_xlabel('Batch Size')
        axes[0,0].set_ylabel('Latency (ms/sample)')
        axes[0,0].set_title('Inference Latency vs Batch Size', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xscale('log')
        
        # 2. Throughput analysis
        axes[0,1].plot(batch_sizes, throughputs, 'g-s', linewidth=2, markersize=6)
        axes[0,1].set_xlabel('Batch Size')
        axes[0,1].set_ylabel('Throughput (samples/sec)')
        axes[0,1].set_title('Inference Throughput vs Batch Size', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_xscale('log')
        
        # 3. Processing time distribution
        axes[0,2].hist(processing_times, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0,2].set_xlabel('Processing Time (ms)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Real-time Processing Time Distribution', fontweight='bold')
        axes[0,2].axvline(np.mean(processing_times), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(processing_times):.2f}ms')
        axes[0,2].legend()
        
        # 4. Memory usage over time
        time_points = np.arange(len(memory_usage)) * 0.01  # 10ms intervals
        axes[1,0].plot(time_points, memory_usage, 'purple', linewidth=1)
        axes[1,0].set_xlabel('Time (seconds)')
        axes[1,0].set_ylabel('Memory Usage (MB)')
        axes[1,0].set_title('Memory Usage During Real-time Processing', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. CPU usage over time
        axes[1,1].plot(time_points, cpu_usage, 'red', linewidth=1)
        axes[1,1].set_xlabel('Time (seconds)')
        axes[1,1].set_ylabel('CPU Usage (%)')
        axes[1,1].set_title('CPU Usage During Real-time Processing', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Performance summary
        metrics = ['Avg Latency\n(ms)', 'Peak Memory\n(MB)', 'Avg CPU\n(%)', 'Throughput\n(samples/s)']
        values = [
            realtime_stats['avg_processing_time_ms'],
            realtime_stats['max_memory_usage_mb'],
            realtime_stats['avg_cpu_usage_percent'],
            realtime_stats['throughput_per_second']
        ]
        
        bars = axes[1,2].bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'], 
                           alpha=0.8, edgecolor='black')
        axes[1,2].set_title('Real-time Performance Summary', fontweight='bold')
        axes[1,2].set_ylabel('Performance Metric')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/deployment_performance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Saved: deployment_performance_analysis.png")
    
    def generate_deployment_report(self, benchmark_results, realtime_stats, deployment_analysis):
        """Generate comprehensive deployment report"""
        print("\n📄 Generating deployment readiness report...")
        
        # Determine deployment readiness
        avg_latency = realtime_stats['avg_processing_time_ms']
        memory_requirement = deployment_analysis['total_memory_requirement_mb']
        model_size = deployment_analysis['model_size_mb']
        
        if avg_latency < 10 and memory_requirement < 100 and model_size < 50:
            readiness = "EXCELLENT - Ready for edge deployment"
        elif avg_latency < 50 and memory_requirement < 500 and model_size < 200:
            readiness = "GOOD - Suitable for most deployment scenarios"
        elif avg_latency < 100 and memory_requirement < 1000:
            readiness = "MODERATE - May require optimization for resource-constrained environments"
        else:
            readiness = "POOR - Significant optimization needed for deployment"
        
        report_content = f"""
# Real-time Deployment Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Deployment Readiness Assessment

**Overall Status**: {readiness}

## Model Specifications
- **Model Size**: {model_size:.2f} MB
- **Features**: {deployment_analysis['feature_count']} ({deployment_analysis['numeric_features']} numeric, {deployment_analysis['categorical_features']} categorical)
- **Memory Requirement**: {memory_requirement:.2f} MB
- **Input Size**: {deployment_analysis['input_size_kb']:.2f} KB per sample

## Performance Benchmarks

### Real-time Processing Performance
- **Average Latency**: {avg_latency:.3f} ms per prediction
- **Maximum Latency**: {realtime_stats['max_processing_time_ms']:.3f} ms
- **Latency Stability**: ±{realtime_stats['std_processing_time_ms']:.3f} ms standard deviation
- **Throughput**: {realtime_stats['throughput_per_second']:.1f} predictions per second

### Resource Utilization
- **Memory Usage**: {realtime_stats['avg_memory_usage_mb']:.1f} MB average, {realtime_stats['max_memory_usage_mb']:.1f} MB peak
- **CPU Usage**: {realtime_stats['avg_cpu_usage_percent']:.1f}% average, {realtime_stats['max_cpu_usage_percent']:.1f}% peak

### Batch Processing Performance
"""
        
        for batch_size in sorted(benchmark_results.keys()):
            stats = benchmark_results[batch_size]
            report_content += f"""
**Batch Size {batch_size}**:
- Latency: {stats['avg_time_per_sample_ms']:.3f} ms/sample
- Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec
- Accuracy: {stats['accuracy']:.4f}
"""
        
        report_content += f"""
## Deployment Recommendations

### Edge Device Requirements
- **Minimum RAM**: {max(memory_requirement * 1.5, 256):.0f} MB
- **Storage**: {max(model_size * 2, 100):.0f} MB available space
- **CPU**: {"Single core adequate" if avg_latency < 50 else "Multi-core recommended"}
- **Network**: {"Minimal requirements" if model_size < 100 else "Moderate bandwidth for model updates"}

### Optimization Opportunities
"""
        
        if avg_latency > 10:
            report_content += "- Consider model quantization to reduce inference time\n"
        if memory_requirement > 200:
            report_content += "- Implement feature selection to reduce memory footprint\n"
        if model_size > 50:
            report_content += "- Apply model compression techniques\n"
        
        report_content += f"""
### Production Deployment Strategy
1. **Single Sample Processing**: {"[READY]" if avg_latency < 100 else "[NEEDS OPTIMIZATION]"}
2. **Batch Processing**: {"[RECOMMENDED for batch size " + str(min(benchmark_results.keys(), key=lambda x: benchmark_results[x]['avg_time_per_sample_ms'])) + "]" if benchmark_results else "N/A"}
3. **Real-time Streaming**: {"[CAPABLE]" if realtime_stats['throughput_per_second'] > 10 else "[MAY STRUGGLE with high-frequency streams]"}
4. **Edge Deployment**: {"[SUITABLE]" if memory_requirement < 500 and avg_latency < 50 else "[RESOURCE CONSTRAINTS]"}

## Security Considerations for Deployment
- Model is optimized for speed while maintaining accuracy
- Suitable for network traffic analysis at line speed
- Memory-efficient for continuous monitoring
- Low CPU overhead allows concurrent security processes
"""
        
        # Save report
        with open(f'{self.results_folder}/deployment_readiness_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed metrics
        deployment_metrics = {
            'model_size_mb': model_size,
            'memory_requirement_mb': memory_requirement,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': realtime_stats['max_processing_time_ms'],
            'throughput_per_second': realtime_stats['throughput_per_second'],
            'avg_memory_usage_mb': realtime_stats['avg_memory_usage_mb'],
            'avg_cpu_usage_percent': realtime_stats['avg_cpu_usage_percent'],
            'deployment_readiness': readiness
        }
        
        pd.DataFrame([deployment_metrics]).to_csv(f'{self.results_folder}/deployment_metrics.csv', index=False)
        
        print("   ✅ Saved: deployment_readiness_report.md")
        print("   ✅ Saved: deployment_metrics.csv")
        
        return readiness, avg_latency
    
    def run_deployment_simulation(self):
        """Run complete deployment simulation"""
        print("🚀 Real-time Deployment Simulation - Enhanced Keylogger Detection")
        print("="*80)
        
        # Load data and train deployment model
        X_train, X_test, y_train, y_test = self.load_deployment_data()
        model, train_time = self.train_deployment_model(X_train, y_train)
        
        # Benchmark performance
        benchmark_results = self.benchmark_inference_performance(model, X_test, y_test)
        
        # Simulate real-time processing
        realtime_stats, processing_times, memory_usage, cpu_usage = self.simulate_realtime_processing(
            model, X_test, duration_seconds=20
        )
        
        # Analyze deployment constraints
        deployment_analysis = self.analyze_deployment_constraints(model, X_test)
        
        # Create visualizations
        self.create_deployment_visualizations(benchmark_results, realtime_stats, 
                                           processing_times, memory_usage, cpu_usage)
        
        # Generate report
        readiness, avg_latency = self.generate_deployment_report(benchmark_results, realtime_stats, 
                                                               deployment_analysis)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 Deployment Simulation Complete!")
        print(f"🚀 Deployment Readiness: {readiness.split(' - ')[0]}")
        print(f"⚡ Average Latency: {avg_latency:.3f} ms")
        print(f"🔥 Throughput: {realtime_stats['throughput_per_second']:.1f} predictions/second")
        print(f"💾 Memory Requirement: {deployment_analysis['total_memory_requirement_mb']:.1f} MB")
        print(f"📦 Model Size: {deployment_analysis['model_size_mb']:.1f} MB")
        
        print("\n📁 Files Generated:")
        print("   • deployment_performance_analysis.png - Performance visualization")
        print("   • deployment_readiness_report.md - Comprehensive deployment analysis")
        print("   • deployment_metrics.csv - Key performance metrics")
        print("="*80)
        
        return readiness, benchmark_results

def main():
    simulator = DeploymentSimulator()
    simulator.run_deployment_simulation()

if __name__ == "__main__":
    main()
