#!/usr/bin/env python3
"""
Python training script that loads data and architecture exported from Node.js
and trains the model with GPU/Metal acceleration on Apple Silicon Macs.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Metal Available:", len(tf.config.list_physical_devices('GPU')) > 0)

class PythonLLMTrainer:
    def __init__(self):
        self.model = None
        self.training_data = None
        self.architecture = None
    
    def load_training_data(self, filename='../training_data.json'):
        """Load training data exported from Node.js"""
        print(f"Loading training data from {filename}...")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.training_data = data
        print(f"Loaded {len(data['samples'])} training samples")
        print(f"Input size: {data['metadata']['inputSize']}")
        print(f"Output size: {data['metadata']['outputSize']}")
        
        return data
    
    def load_model_architecture(self, filename='../model_architecture.json'):
        """Load model architecture exported from Node.js"""
        print(f"Loading model architecture from {filename}...")
        
        with open(filename, 'r') as f:
            arch = json.load(f)
        
        self.architecture = arch
        print("Architecture loaded successfully")
        return arch
    
    def prepare_data(self):
        """Convert loaded data to TensorFlow format"""
        if not self.training_data:
            raise ValueError("Training data not loaded. Call load_training_data() first.")
        
        # Extract inputs and targets
        inputs = []
        targets = []
        
        for sample in self.training_data['samples']:
            # Normalize inputs (1-9 -> 0-1 range)
            normalized_input = [(x - 1) / 8 for x in sample['input']]
            inputs.append(normalized_input)
            
            # Target is 1-9, convert to 0-8 for sparse categorical crossentropy
            targets.append(sample['target'] - 1)
        
        # Convert to numpy arrays
        X = np.array(inputs, dtype=np.float32)
        y = np.array(targets, dtype=np.int32)
        
        print(f"Prepared data shapes: X={X.shape}, y={y.shape}")
        return X, y
    
    def build_model(self):
        """Build model from loaded architecture"""
        if not self.architecture:
            raise ValueError("Architecture not loaded. Call load_model_architecture() first.")
        
        arch = self.architecture
        layers = arch['layers']
        
        model = tf.keras.Sequential()
        
        for i, layer_config in enumerate(layers):
            if layer_config['type'] == 'Dense':
                if i == 0:  # First layer needs input shape
                    model.add(tf.keras.layers.Dense(
                        units=layer_config['units'],
                        activation=layer_config['activation'],
                        input_shape=layer_config['inputShape']
                    ))
                else:
                    model.add(tf.keras.layers.Dense(
                        units=layer_config['units'],
                        activation=layer_config['activation']
                    ))
        
        # Compile model
        compile_config = arch['compile']
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=compile_config['learningRate']),
            loss=compile_config['loss'],
            metrics=compile_config['metrics']
        )
        
        self.model = model
        print("Model built successfully!")
        model.summary()
        return model
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with GPU acceleration"""
        if not self.model:
            raise ValueError("Model not built. Call build_model() first.")
        
        X, y = self.prepare_data()
        
        print(f"\nStarting training with {epochs} epochs...")
        print(f"Using device: {tf.config.list_physical_devices()}")
        
        # Train with GPU/Metal acceleration
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def save_model(self, filename='../python_trained_model.h5'):
        """Save the trained model"""
        if not self.model:
            raise ValueError("Model not trained yet.")
        
        self.model.save(filename)
        print(f"Model saved to {filename}")
    
    def test_predictions(self):
        """Test the model with the same test cases as Node.js"""
        if not self.model:
            raise ValueError("Model not trained yet.")
        
        test_cases = [
            [1, 2, 3, 4, 5],  # Arithmetic +1
            [2, 4, 6, 8, 1],  # Arithmetic +2 (wrapping) 
            [1, 1, 2, 3, 5],  # Fibonacci-like
            [1, 2, 1, 2, 1],  # Repeating pattern
            [5, 5, 5, 5, 5],  # Constant
            [9, 8, 7, 6, 5],  # Arithmetic -1
            [1, 3, 5, 7, 9]   # Arithmetic +2
        ]
        
        print("\n=== Testing Python-trained model ===")
        
        for i, test_case in enumerate(test_cases):
            # Normalize input
            normalized = [(x - 1) / 8 for x in test_case]
            X_test = np.array([normalized], dtype=np.float32)
            
            # Predict
            predictions = self.model.predict(X_test, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            predicted_number = predicted_class + 1  # Convert back to 1-9
            
            print(f"Test {i+1}: {test_case} -> {predicted_number} (confidence: {confidence*100:.1f}%)")

def main():
    # Check if data files exist
    if not Path('../training_data.json').exists():
        print("Error: training_data.json not found!")
        print("Run 'npm run train' first to generate the data.")
        return
    
    if not Path('../model_architecture.json').exists():
        print("Error: model_architecture.json not found!")
        print("Run 'npm run train' first to generate the architecture.")
        return
    
    # Create trainer and run
    trainer = PythonLLMTrainer()
    
    # Load data and architecture from Node.js exports
    trainer.load_training_data()
    trainer.load_model_architecture()
    
    # Build and train model
    trainer.build_model()
    history = trainer.train(epochs=100)  # More epochs for better training
    
    # Test and save
    trainer.test_predictions()
    trainer.save_model()
    
    print(f"\n🎉 Python training complete!")
    print(f"Final accuracy: {max(history.history['accuracy']):.3f}")
    print(f"Final validation accuracy: {max(history.history['val_accuracy']):.3f}")

if __name__ == "__main__":
    main()