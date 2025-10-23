#!/usr/bin/env python3
"""
CPU vs GPU Performance Comparison Test
Direct comparison with identical training conditions
"""

import json
import time
import numpy as np
import tensorflow as tf

print("🚀 CPU vs GPU Performance Comparison")
print("=" * 50)

def load_data():
    """Load training data"""
    print("Loading training data...")
    with open('../training_data.json', 'r') as f:
        data = json.load(f)
    
    # Prepare data (same as trainer)
    inputs = []
    targets = []
    
    for sample in data['samples']:
        normalized_input = [(x - 1) / 8 for x in sample['input']]
        inputs.append(normalized_input)
        targets.append(sample['target'] - 1)  # 0-8 for sparse categorical
    
    X = np.array(inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.int32)
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    return X, y

def create_model():
    """Create the same model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_on_device(device_name, X, y, epochs=10):
    """Train model on specified device and measure performance"""
    print(f"\n🔥 Training on {device_name}")
    print("-" * 30)
    
    with tf.device(device_name):
        # Create fresh model
        model = create_model()
        
        print(f"Device: {device_name}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(X)}")
        
        # Measure training time
        start_time = time.time()
        
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1  # Show progress
        )
        
        training_time = time.time() - start_time
        
        # Get final metrics
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        final_loss = history.history['loss'][-1]
        
        print(f"\n📊 Results for {device_name}:")
        print(f"  Total training time: {training_time:.2f} seconds")
        print(f"  Time per epoch: {training_time/epochs:.2f} seconds")
        print(f"  Final accuracy: {final_accuracy:.3f}")
        print(f"  Final val accuracy: {final_val_accuracy:.3f}")
        print(f"  Final loss: {final_loss:.3f}")
        
        return {
            'device': device_name,
            'total_time': training_time,
            'time_per_epoch': training_time / epochs,
            'final_accuracy': final_accuracy,
            'final_val_accuracy': final_val_accuracy,
            'final_loss': final_loss,
            'history': history.history
        }

def run_comparison():
    """Run the full CPU vs GPU comparison"""
    
    # Check available devices
    print("Available devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  - {device}")
    
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    # Load data once
    X, y = load_data()
    
    # Training parameters
    epochs = 15  # Fewer epochs for quick comparison
    
    results = {}
    
    # Test CPU
    print(f"\n{'='*50}")
    results['cpu'] = train_on_device('/CPU:0', X, y, epochs)
    
    # Test GPU if available
    if gpu_available:
        print(f"\n{'='*50}")
        results['gpu'] = train_on_device('/GPU:0', X, y, epochs)
    else:
        print("\n❌ No GPU available for comparison")
        return results
    
    # Comparison summary
    print(f"\n{'='*50}")
    print("🏆 PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    cpu_time = results['cpu']['total_time']
    gpu_time = results['gpu']['total_time']
    speedup = cpu_time / gpu_time
    
    print(f"\n⏱️  Training Time ({epochs} epochs):")
    print(f"  CPU: {cpu_time:.2f} seconds")
    print(f"  GPU: {gpu_time:.2f} seconds")
    print(f"  🚀 GPU Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
    
    print(f"\n⏱️  Time Per Epoch:")
    print(f"  CPU: {results['cpu']['time_per_epoch']:.2f} seconds")
    print(f"  GPU: {results['gpu']['time_per_epoch']:.2f} seconds")
    
    print(f"\n🎯 Final Accuracy:")
    print(f"  CPU: {results['cpu']['final_accuracy']:.3f}")
    print(f"  GPU: {results['gpu']['final_accuracy']:.3f}")
    
    print(f"\n🎯 Final Validation Accuracy:")
    print(f"  CPU: {results['cpu']['final_val_accuracy']:.3f}")
    print(f"  GPU: {results['gpu']['final_val_accuracy']:.3f}")
    
    # Performance analysis
    print(f"\n📈 Analysis:")
    if speedup > 1.5:
        print(f"  ✅ Excellent GPU acceleration ({speedup:.1f}x speedup)")
    elif speedup > 1.1:
        print(f"  ✅ Good GPU acceleration ({speedup:.1f}x speedup)")
    elif speedup > 0.9:
        print(f"  ⚠️  Similar performance (GPU overhead for small models)")
    else:
        print(f"  ❌ GPU slower (model too small for GPU advantage)")
    
    # Training efficiency
    cpu_samples_per_sec = len(X) * epochs / cpu_time
    gpu_samples_per_sec = len(X) * epochs / gpu_time
    
    print(f"\n🔥 Training Efficiency:")
    print(f"  CPU: {cpu_samples_per_sec:.0f} samples/second")
    print(f"  GPU: {gpu_samples_per_sec:.0f} samples/second")
    
    return results

def quick_matrix_test():
    """Quick matrix multiplication test to verify GPU is working"""
    print(f"\n{'='*50}")
    print("🧮 Quick Matrix Multiplication Test")
    print(f"{'='*50}")
    
    size = 2000
    print(f"Testing {size}x{size} matrix multiplication...")
    
    # CPU test
    with tf.device('/CPU:0'):
        a_cpu = tf.random.normal([size, size])
        b_cpu = tf.random.normal([size, size])
        
        start = time.time()
        c_cpu = tf.matmul(a_cpu, b_cpu)
        _ = c_cpu.numpy()  # Force execution
        cpu_time = time.time() - start
    
    # GPU test
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        with tf.device('/GPU:0'):
            a_gpu = tf.random.normal([size, size])
            b_gpu = tf.random.normal([size, size])
            
            start = time.time()
            c_gpu = tf.matmul(a_gpu, b_gpu)
            _ = c_gpu.numpy()  # Force execution
            gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"  CPU: {cpu_time:.3f} seconds")
        print(f"  GPU: {gpu_time:.3f} seconds")
        print(f"  🚀 Matrix speedup: {speedup:.1f}x")
    else:
        print("  No GPU available")

if __name__ == "__main__":
    try:
        # Quick matrix test first
        quick_matrix_test()
        
        # Full neural network comparison
        results = run_comparison()
        
        print(f"\n🎉 Comparison complete!")
        print(f"Results show the performance difference between CPU and GPU training.")
        
    except FileNotFoundError:
        print("❌ training_data.json not found!")
        print("Run 'npm run train' first to generate training data.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()