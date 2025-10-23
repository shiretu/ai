#!/usr/bin/env python3
"""
CPU vs GPU Performance Comparison with Different Model Sizes
Shows how model size affects GPU advantage
"""

import json
import time
import numpy as np
import tensorflow as tf

print("🚀 CPU vs GPU Performance Comparison - Multiple Model Sizes")
print("=" * 60)

def load_data():
    """Load training data"""
    print("Loading training data...")
    with open('training_data.json', 'r') as f:
        data = json.load(f)
    
    inputs = []
    targets = []
    
    for sample in data['samples']:
        normalized_input = [(x - 1) / 8 for x in sample['input']]
        inputs.append(normalized_input)
        targets.append(sample['target'] - 1)
    
    X = np.array(inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.int32)
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    return X, y

def create_model(size="small"):
    """Create models of different sizes"""
    if size == "small":
        # Original small model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])
        params = "~1.5K"
    
    elif size == "medium":
        # Medium model - better for GPU
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])
        params = "~25K"
    
    elif size == "large":
        # Large model - should definitely benefit from GPU
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])
        params = "~400K"
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, params

def train_model_comparison(X, y, model_size, epochs=10):
    """Compare CPU vs GPU for a specific model size"""
    print(f"\n{'='*60}")
    print(f"🧠 Testing {model_size.upper()} Model")
    print(f"{'='*60}")
    
    results = {}
    
    # Get model info
    model_cpu, params = create_model(model_size)
    model_cpu.build(input_shape=(None, 5))
    total_params = model_cpu.count_params()
    
    print(f"Model size: {params} parameters ({total_params:,} exact)")
    print(f"Epochs: {epochs}")
    
    # CPU Training
    print(f"\n🔥 CPU Training ({model_size})")
    print("-" * 30)
    
    with tf.device('/CPU:0'):
        model_cpu, _ = create_model(model_size)
        
        start_time = time.time()
        history_cpu = model_cpu.fit(
            X, y,
            epochs=epochs,
            batch_size=64,  # Larger batch for better comparison
            validation_split=0.2,
            verbose=0  # Silent for cleaner output
        )
        cpu_time = time.time() - start_time
        
        cpu_accuracy = history_cpu.history['accuracy'][-1]
        cpu_val_accuracy = history_cpu.history['val_accuracy'][-1]
    
    results['cpu'] = {
        'time': cpu_time,
        'accuracy': cpu_accuracy,
        'val_accuracy': cpu_val_accuracy,
        'samples_per_sec': len(X) * epochs / cpu_time
    }
    
    print(f"  Time: {cpu_time:.2f} seconds")
    print(f"  Final accuracy: {cpu_accuracy:.3f}")
    print(f"  Samples/sec: {results['cpu']['samples_per_sec']:.0f}")
    
    # GPU Training
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"\n🚀 GPU Training ({model_size})")
        print("-" * 30)
        
        with tf.device('/GPU:0'):
            model_gpu, _ = create_model(model_size)
            
            start_time = time.time()
            history_gpu = model_gpu.fit(
                X, y,
                epochs=epochs,
                batch_size=64,
                validation_split=0.2,
                verbose=0
            )
            gpu_time = time.time() - start_time
            
            gpu_accuracy = history_gpu.history['accuracy'][-1]
            gpu_val_accuracy = history_gpu.history['val_accuracy'][-1]
        
        results['gpu'] = {
            'time': gpu_time,
            'accuracy': gpu_accuracy,
            'val_accuracy': gpu_val_accuracy,
            'samples_per_sec': len(X) * epochs / gpu_time
        }
        
        speedup = cpu_time / gpu_time
        
        print(f"  Time: {gpu_time:.2f} seconds")
        print(f"  Final accuracy: {gpu_accuracy:.3f}")
        print(f"  Samples/sec: {results['gpu']['samples_per_sec']:.0f}")
        print(f"  🚀 Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        
        results['speedup'] = speedup
    
    return results

def run_comprehensive_comparison():
    """Run comparison across different model sizes"""
    
    # Load data
    X, y = load_data()
    
    # Test parameters
    epochs = 8  # Reduced for faster testing
    
    # Test different model sizes
    model_sizes = ["small", "medium", "large"]
    all_results = {}
    
    for size in model_sizes:
        all_results[size] = train_model_comparison(X, y, size, epochs)
    
    # Summary
    print(f"\n{'='*60}")
    print("🏆 COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    gpu_available = 'gpu' in all_results['small']
    
    if gpu_available:
        print(f"\n{'Model Size':<10} {'CPU Time':<10} {'GPU Time':<10} {'Speedup':<10} {'Winner':<10}")
        print("-" * 55)
        
        for size in model_sizes:
            r = all_results[size]
            cpu_t = r['cpu']['time']
            gpu_t = r['gpu']['time']
            speedup = r['speedup']
            winner = "🚀 GPU" if speedup > 1 else "🔥 CPU"
            
            print(f"{size.capitalize():<10} {cpu_t:<10.2f} {gpu_t:<10.2f} {speedup:<10.1f}x {winner:<10}")
        
        # Analysis
        print(f"\n📊 Analysis:")
        best_speedup_size = max(model_sizes, key=lambda s: all_results[s].get('speedup', 0))
        best_speedup = all_results[best_speedup_size]['speedup']
        
        print(f"  • Best GPU speedup: {best_speedup:.1f}x with {best_speedup_size} model")
        
        if best_speedup > 1.5:
            print(f"  ✅ GPU shows clear advantage with larger models")
        else:
            print(f"  ⚠️  GPU overhead dominates with this model size/data size combination")
        
        print(f"\n💡 Insights:")
        print(f"  • Small models: CPU faster due to GPU overhead")
        print(f"  • Medium models: GPU starts to show benefit")
        print(f"  • Large models: GPU should show clear advantage")
        print(f"  • With larger datasets, GPU advantage would be more pronounced")
    
    else:
        print("❌ No GPU available for comparison")
    
    return all_results

def quick_tensor_ops_test():
    """Test different tensor operations to show GPU strengths"""
    print(f"\n{'='*60}")
    print("🧮 Raw Tensor Operations Comparison")
    print(f"{'='*60}")
    
    operations = [
        ("Matrix Mult 1000x1000", lambda: tf.matmul(tf.random.normal([1000, 1000]), tf.random.normal([1000, 1000]))),
        ("Matrix Mult 2000x2000", lambda: tf.matmul(tf.random.normal([2000, 2000]), tf.random.normal([2000, 2000]))),
        ("Conv2D Operation", lambda: tf.nn.conv2d(tf.random.normal([32, 64, 64, 3]), tf.random.normal([3, 3, 3, 32]), strides=1, padding='SAME')),
    ]
    
    for op_name, op_func in operations:
        print(f"\n{op_name}:")
        
        # CPU
        with tf.device('/CPU:0'):
            start = time.time()
            result_cpu = op_func()
            _ = result_cpu.numpy()
            cpu_time = time.time() - start
        
        # GPU
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            with tf.device('/GPU:0'):
                start = time.time()
                result_gpu = op_func()
                _ = result_gpu.numpy()
                gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time
            print(f"  CPU: {cpu_time:.3f}s | GPU: {gpu_time:.3f}s | Speedup: {speedup:.1f}x")
        else:
            print(f"  CPU: {cpu_time:.3f}s | GPU: Not available")

if __name__ == "__main__":
    try:
        # Quick tensor ops test
        quick_tensor_ops_test()
        
        # Comprehensive neural network comparison
        results = run_comprehensive_comparison()
        
        print(f"\n🎉 Testing complete!")
        print(f"This shows how model size affects CPU vs GPU performance.")
        
    except FileNotFoundError:
        print("❌ training_data.json not found!")
        print("Run 'npm run train' first to generate training data.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()