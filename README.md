# AI Number Sequence Prediction

A minimalistic LLM that predicts the next number in sequences (1-9). Features both Node.js prototyping and Python high-performance training.

## 🚀 Quick Start

### Node.js Development (Prototyping)
```bash
# Install dependencies
npm install

# Train and export data
npm run train
```

### Python Training (High Performance)
```bash
# Setup Python environment (one time)
npm run setup-python

# Train with GPU/Metal acceleration
npm run train-python

# Compare CPU vs GPU performance
npm run compare-cpu-gpu
npm run compare-comprehensive
```

## 📊 Workflow: Node.js → Python

This project demonstrates a powerful workflow:

1. **🔬 Prototype in Node.js** - Easy data generation and network design
2. **⚡ Train in Python** - GPU/Metal acceleration for serious training
3. **🔄 Transfer Models** - Move trained weights between environments

### Node.js (Prototyping)
- ✅ Easy data generation with multiple patterns
- ✅ Quick network architecture experimentation  
- ✅ Familiar JavaScript syntax
- ✅ Export data and architecture for Python
- ❌ CPU-only training (slower)

### Python (Production Training)
- ✅ GPU/Metal acceleration on Apple Silicon
- ✅ TensorFlow's full ecosystem
- ✅ Faster training and larger models
- ✅ Production-ready model saving
- ❌ Requires Python knowledge for modifications

## 🏗️ Architecture

**Neural Network:**
- Input: 5 numbers (normalized 0-1)
- Hidden: 2 layers × 32 nodes (ReLU)
- Output: 9 nodes (softmax for digits 1-9)
- Total: 1,545 parameters

**Training Patterns:**
- Arithmetic sequences: `[1,2,3,4,5] → 6`
- Repeating patterns: `[1,2,1,2,1] → 2` 
- Fibonacci-like: `[1,1,2,3,5] → 8`
- Biased random sequences

## 📁 Project Structure

```
├── src/                    # Node.js source files
│   ├── llm.js             # Main LLM implementation
│   ├── test.js            # Basic tests
│   └── check-backend.js   # TensorFlow.js backend check
├── python/                # Python training environment
│   ├── python_trainer.py         # Main GPU trainer
│   ├── cpu_vs_gpu_test.py        # Performance comparison
│   ├── cpu_vs_gpu_comprehensive.py # Multi-size comparison
│   ├── python_hw_test.py          # Hardware diagnostics
│   ├── python_quick_check.py      # Quick GPU check
│   ├── gpu_diagnostics.py         # Detailed GPU info
│   ├── requirements.txt           # Python dependencies
│   └── venv/                      # Virtual environment (auto-created)
├── training_data.json      # Generated training samples
├── model_architecture.json # Network structure for Python
├── python_trained_model.h5 # Saved trained model
└── package.json           # Node.js scripts and dependencies
```

## 📋 Available Scripts

**Node.js:**
- `npm run train` - Generate data and train in Node.js
- `npm test` - Run basic tests

**Python Setup:**
- `npm run setup-python` - Create Python environment and install dependencies

**Python Training:**
- `npm run train-python` - Train with GPU acceleration
- `npm run check-python` - Quick hardware/GPU check
- `npm run test-python-hw` - Detailed hardware diagnostics

**Performance Testing:**
- `npm run compare-cpu-gpu` - Basic CPU vs GPU comparison
- `npm run compare-comprehensive` - Multi-model-size comparison

## 📊 Files Generated

After running `npm run train`:

- `training_data.json` - 10,000 training samples with patterns
- `model_architecture.json` - Network structure for Python
- Node.js trains and exports everything automatically

## 🧪 Testing

**Node.js Results:**
```
Test 1: [1, 2, 3, 4, 5] -> 6 (confidence: 90.7%)
Test 3: [1, 1, 2, 3, 5] -> 8 (confidence: 89.4%)
Test 7: [1, 3, 5, 7, 9] -> 2 (confidence: 92.3%)
```

**Python Results:** (with more training epochs)
- Higher accuracy with GPU acceleration
- Same network architecture
- Same test cases for comparison

## 🔧 Customization

### Add New Patterns (Node.js)
```javascript
// In src/llm.js, add to generateTrainingData()
case 4: // Your new pattern
    sequence = this.generateCustomPattern()
    break
```

### Modify Architecture (Node.js)
```javascript
// In src/llm.js constructor
this.hiddenSize = 64  // More neurons
// Architecture auto-exports to Python
```

### Python Training Options
```python
# In python/python_trainer.py
trainer.train(
    epochs=200,      # More training
    batch_size=64,   # Larger batches
    validation_split=0.3
)
```

## 🧪 Performance Analysis

The project includes comprehensive CPU vs GPU comparison tools that show:

- **Small models (1.5K params)**: CPU is 5x faster due to GPU overhead
- **Medium models (26K params)**: CPU still wins, but gap narrows  
- **Large models (431K+ params)**: GPU starts to show advantage
- **Matrix operations**: GPU excels at 2000x2000+ operations (2.3x speedup)

**Key Insight**: For small training datasets and models, CPU is often faster! GPU shines with larger workloads.

## 🎯 Use Cases

This workflow is perfect for:
- **Rapid prototyping** of sequence models
- **Educational ML projects** 
- **Pattern recognition experiments**
- **Learning LLM fundamentals**
- **Testing ideas before big training runs**

## 📋 Requirements

**Node.js:**
- Node.js 16+
- @tensorflow/tfjs

**Python (optional):**
- Python 3.8+
- TensorFlow 2.13+
- NumPy

## 🤖 Next Steps

- Scale to larger vocabularies
- Add more complex patterns
- Experiment with different architectures
- Transfer to text prediction
- Add attention mechanisms

Perfect for learning how LLMs work under the hood! 🎓