#!/usr/bin/env python3
"""
Python Hardware Acceleration Tester
Tests TensorFlow GPU/Metal support on Apple Silicon Macs
"""

import sys
import time
import numpy as np

def check_python_setup():
    """Check Python and basic imports"""
    print("=== Python Setup ===")
    print(f"Python version: {sys.version}")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        return tf
    except ImportError:
        print("❌ TensorFlow not installed")
        print("Run: pip3 install tensorflow")
        return None
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        print("Run: pip3 install numpy")
        return None

def check_hardware_acceleration(tf):
    """Check what hardware acceleration is available"""
    print("\n=== Hardware Acceleration Check ===")
    
    # Check physical devices
    physical_devices = tf.config.list_physical_devices()
    print(f"Physical devices: {len(physical_devices)}")
    for device in physical_devices:
        print(f"  - {device}")
    
    # Check GPU devices specifically
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices: {len(gpu_devices)}")
    for gpu in gpu_devices:
        print(f"  - {gpu}")
    
    # Check if Metal is available (Apple Silicon)
    if len(gpu_devices) > 0:
        print("✅ GPU acceleration available!")
        if sys.platform == 'darwin':
            print("✅ Running on macOS - likely Metal acceleration")
    else:
        print("❌ No GPU devices found")
    
    # Check logical devices
    logical_devices = tf.config.list_logical_devices()
    print(f"\nLogical devices: {len(logical_devices)}")
    for device in logical_devices:
        print(f"  - {device}")
    
    return len(gpu_devices) > 0

def performance_benchmark(tf, use_gpu=True):
    """Run performance benchmark to test actual acceleration"""
    print("\n=== Performance Benchmark ===")
    
    # Configure device
    if use_gpu and len(tf.config.list_physical_devices('GPU')) > 0:
        device_name = '/GPU:0'
        print("Testing with GPU acceleration...")
    else:
        device_name = '/CPU:0'
        print("Testing with CPU only...")
    
    # Test different matrix sizes
    sizes = [500, 1000, 2000]
    results = {}
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrix multiplication...")
        
        with tf.device(device_name):
            # Create random matrices
            a = tf.random.normal([size, size], dtype=tf.float32)
            b = tf.random.normal([size, size], dtype=tf.float32)
            
            # Warm up
            _ = tf.matmul(a, b)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):  # Multiple runs for stability
                result = tf.matmul(a, b)
                # Force computation to complete
                _ = result.numpy()
            elapsed = (time.time() - start_time) / 5
            
            results[size] = elapsed
            print(f"  Average time: {elapsed:.3f}s")
            print(f"  Operations/sec: {(size*size*size*2) / elapsed / 1e9:.2f} GFLOPS")
    
    return results

def test_neural_network_training(tf):
    """Test actual neural network training performance"""
    print("\n=== Neural Network Training Test ===")
    
    # Create a simple model similar to our LLM
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Generate test data
    print("\nGenerating test data...")
    X = np.random.random((1000, 5)).astype(np.float32)
    y = np.random.randint(0, 9, (1000,)).astype(np.int32)
    
    # Training benchmark
    print("\nTraining benchmark (10 epochs)...")
    start_time = time.time()
    
    history = model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Time per epoch: {training_time/10:.2f} seconds")
    
    return training_time

def memory_usage_test(tf):
    """Test GPU memory usage"""
    print("\n=== Memory Usage Test ===")
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        try:
            # Get GPU memory info
            gpu_details = tf.config.experimental.get_device_details(
                tf.config.list_logical_devices('GPU')[0]
            )
            print("GPU details:", gpu_details)
        except:
            print("Could not get detailed GPU information")
        
        # Test memory allocation
        print("\nTesting memory allocation...")
        with tf.device('/GPU:0'):
            # Allocate progressively larger tensors
            sizes = [100, 500, 1000, 2000]
            for size in sizes:
                try:
                    tensor = tf.random.normal([size, size])
                    print(f"✅ Successfully allocated {size}x{size} tensor")
                    del tensor  # Clean up
                except Exception as e:
                    print(f"❌ Failed to allocate {size}x{size} tensor: {e}")
                    break
    else:
        print("No GPU available for memory testing")

def compare_cpu_vs_gpu(tf):
    """Direct comparison between CPU and GPU performance"""
    print("\n=== CPU vs GPU Comparison ===")
    
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("No GPU available for comparison")
        return
    
    test_size = 1000
    
    # CPU test
    print(f"CPU test ({test_size}x{test_size} matrix multiplication)...")
    with tf.device('/CPU:0'):
        a = tf.random.normal([test_size, test_size])
        b = tf.random.normal([test_size, test_size])
        
        start_time = time.time()
        result_cpu = tf.matmul(a, b)
        _ = result_cpu.numpy()
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.3f} seconds")
    
    # GPU test
    print(f"GPU test ({test_size}x{test_size} matrix multiplication)...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([test_size, test_size])
        b = tf.random.normal([test_size, test_size])
        
        start_time = time.time()
        result_gpu = tf.matmul(a, b)
        _ = result_gpu.numpy()
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.3f} seconds")
    
    # Calculate speedup
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU Speedup: {speedup:.1f}x faster than CPU")
        if speedup > 2:
            print("✅ Excellent GPU acceleration!")
        elif speedup > 1.2:
            print("✅ Good GPU acceleration")
        else:
            print("⚠️  Limited GPU acceleration - check drivers/setup")
    
    return cpu_time, gpu_time

def main():
    print("🔍 Python Hardware Acceleration Tester")
    print("=====================================")
    
    # Check basic setup
    tf = check_python_setup()
    if not tf:
        return
    
    # Check hardware
    has_gpu = check_hardware_acceleration(tf)
    
    # Performance tests
    if has_gpu:
        print("\n🎯 Running GPU performance tests...")
        gpu_results = performance_benchmark(tf, use_gpu=True)
        compare_cpu_vs_gpu(tf)
        memory_usage_test(tf)
    else:
        print("\n⚠️  No GPU found - running CPU-only tests...")
        cpu_results = performance_benchmark(tf, use_gpu=False)
    
    # Neural network training test
    training_time = test_neural_network_training(tf)
    
    # Summary
    print("\n" + "="*50)
    print("🎉 SUMMARY")
    print("="*50)
    
    if has_gpu:
        print("✅ GPU/Metal acceleration: AVAILABLE")
        print("✅ Ready for high-performance training!")
        print("✅ Your LLM training will be significantly faster")
    else:
        print("❌ GPU/Metal acceleration: NOT AVAILABLE")
        print("⚠️  Training will use CPU only (slower)")
        print("💡 Consider installing TensorFlow with Metal support")
    
    print(f"\nNeural network training time: {training_time:.2f}s")
    print("\n🚀 Ready to run: npm run train-python")

if __name__ == "__main__":
    main()