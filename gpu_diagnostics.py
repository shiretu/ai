#!/usr/bin/env python3
"""
Deep GPU Diagnostics for Apple Silicon
Let's figure out why Metal isn't being detected
"""

import sys
import os

def check_system_info():
    print("🔍 DEEP GPU DIAGNOSTICS")
    print("=" * 50)
    
    print("=== System Information ===")
    print(f"Platform: {sys.platform}")
    print(f"Architecture: {os.uname().machine}")
    print(f"Python: {sys.version}")
    
    # Check macOS version
    if sys.platform == 'darwin':
        import subprocess
        try:
            result = subprocess.run(['sw_vers', '-productVersion'], 
                                  capture_output=True, text=True)
            print(f"macOS: {result.stdout.strip()}")
        except:
            print("Could not determine macOS version")

def check_tensorflow_build():
    print("\n=== TensorFlow Build Information ===")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        
        # Check what TensorFlow was built with
        print("\nBuild configuration:")
        build_info = tf.config.list_physical_devices()
        print(f"Physical devices: {build_info}")
        
        # Check GPU support in build
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")
        
        # Check if TensorFlow can see any GPUs
        print(f"GPU available: {tf.test.is_gpu_available()}")
        
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False
    
    return True

def check_metal_support():
    print("\n=== Metal Support Investigation ===")
    
    try:
        import tensorflow as tf
        
        # Try to force Metal backend
        print("Checking for Metal backend...")
        
        # List all available devices in detail
        physical_devices = tf.config.list_physical_devices()
        print(f"All physical devices: {physical_devices}")
        
        # Check specifically for GPU devices
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"GPU devices: {gpu_devices}")
        
        # Check logical devices
        logical_devices = tf.config.list_logical_devices()
        print(f"Logical devices: {logical_devices}")
        
        # Try to get device details
        for device in logical_devices:
            try:
                if 'GPU' in device.name:
                    details = tf.config.experimental.get_device_details(device)
                    print(f"GPU details: {details}")
            except Exception as e:
                print(f"Could not get device details: {e}")
        
        # Check environment variables that might affect GPU
        print("\n=== Environment Variables ===")
        gpu_env_vars = ['CUDA_VISIBLE_DEVICES', 'TF_GPU_ALLOCATOR', 'TF_FORCE_GPU_ALLOW_GROWTH']
        for var in gpu_env_vars:
            value = os.environ.get(var, 'Not set')
            print(f"{var}: {value}")
            
    except Exception as e:
        print(f"❌ Error checking Metal support: {e}")

def test_tensorflow_versions():
    print("\n=== TensorFlow Version Compatibility ===")
    
    # Try different TensorFlow imports
    modules_to_check = [
        'tensorflow',
        'tensorflow.python.client.device_lib',
        'tensorflow.config',
        'tensorflow.config.experimental'
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

def check_device_lib():
    print("\n=== Device Library Check ===")
    
    try:
        import tensorflow as tf
        from tensorflow.python.client import device_lib
        
        print("Local devices:")
        devices = device_lib.list_local_devices()
        for device in devices:
            print(f"  Device: {device.name}")
            print(f"    Type: {device.device_type}")
            print(f"    Memory limit: {device.memory_limit}")
            print(f"    Description: {device.physical_device_desc}")
            print()
            
    except Exception as e:
        print(f"❌ Error checking device library: {e}")

def try_gpu_operation():
    print("\n=== GPU Operation Test ===")
    
    try:
        import tensorflow as tf
        
        # Try to create a tensor on GPU
        print("Attempting GPU operation...")
        
        with tf.device('/CPU:0'):
            a = tf.constant([1.0, 2.0])
            print(f"✅ CPU tensor created: {a}")
        
        # Try GPU if available
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            with tf.device('/GPU:0'):
                b = tf.constant([3.0, 4.0])
                print(f"✅ GPU tensor created: {b}")
        else:
            print("❌ No GPU devices to test")
            
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")

def check_metal_plugin():
    print("\n=== Metal Plugin Investigation ===")
    
    # Check if tensorflow-metal is available
    try:
        import tensorflow_metal
        print(f"✅ tensorflow-metal available: {tensorflow_metal.__version__}")
    except ImportError:
        print("❌ tensorflow-metal not installed")
        print("💡 This might be the issue!")
        
        # Check if we can install it
        print("\nChecking tensorflow-metal availability...")
        import subprocess
        try:
            result = subprocess.run([
                'pip', 'show', 'tensorflow-metal'
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print("tensorflow-metal is installed but not importable")
            else:
                print("tensorflow-metal is not installed")
                
                # Try to find compatible version
                print("Checking for compatible tensorflow-metal...")
                result = subprocess.run([
                    'pip', 'index', 'versions', 'tensorflow-metal'
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Available versions: {result.stdout}")
                else:
                    print("Could not find tensorflow-metal in package index")
                    
        except Exception as e:
            print(f"Error checking tensorflow-metal: {e}")

def main():
    check_system_info()
    
    if not check_tensorflow_build():
        return
    
    check_metal_support()
    test_tensorflow_versions()
    check_device_lib()
    try_gpu_operation()
    check_metal_plugin()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 50)
    
    print("\n💡 POTENTIAL SOLUTIONS:")
    print("1. Install tensorflow-metal (if compatible)")
    print("2. Try different TensorFlow version")
    print("3. Check Apple's official ML framework (MLX)")
    print("4. Use older Python version (3.11 or 3.10)")

if __name__ == "__main__":
    main()