#!/usr/bin/env python3
"""
Quick Python Hardware Check
Simple test to verify TensorFlow and GPU support
"""

def quick_check():
    print("🔍 Quick Python Hardware Check")
    print("="*35)
    
    # Check Python
    import sys
    print(f"Python: {sys.version.split()[0]} ✅")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__} ✅")
        
        # Check devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices: {len(gpus)}")
        
        if len(gpus) > 0:
            print("✅ GPU/Metal acceleration: AVAILABLE")
            print(f"   Devices: {[str(gpu) for gpu in gpus]}")
            
            # Quick performance test
            import time
            print("\n⚡ Quick speed test...")
            
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                
                start = time.time()
                c = tf.matmul(a, b)
                _ = c.numpy()  # Force execution
                gpu_time = time.time() - start
                print(f"   GPU: {gpu_time:.3f}s")
            
            with tf.device('/CPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                
                start = time.time()
                c = tf.matmul(a, b)
                _ = c.numpy()
                cpu_time = time.time() - start
                print(f"   CPU: {cpu_time:.3f}s")
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"   🚀 Speedup: {speedup:.1f}x")
            
        else:
            print("❌ GPU/Metal acceleration: NOT AVAILABLE")
            print("   Training will use CPU only")
        
        print(f"\n🎯 Ready for LLM training!")
        
    except ImportError:
        print("❌ TensorFlow not installed")
        print("   Run: pip3 install tensorflow")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_check()