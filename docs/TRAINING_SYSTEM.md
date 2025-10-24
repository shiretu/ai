# Real-time Trading Network Training System

A WebSocket-based training pipeline for continuous learning with streaming market data.

## Overview

This system allows real-time training of the neural network by streaming datasets through WebSocket connections. Perfect for continuous learning as new market data becomes available.

## Architecture

```
Data Generator → WebSocket Client → Training Server → Neural Network
                                       ↓
                                  Model Storage
```

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Training Server
```bash
npm run training-server
```
Server will start on `ws://localhost:8080`

### 3. Send Training Data
```bash
npm run training-client
```
Runs example client with sample data

## Training Server Features

### ✅ Model Management
- **Auto-load**: Loads existing model from disk if available
- **Auto-create**: Creates new model if none exists
- **Auto-save**: Saves model after training sessions

### ✅ Real-time Training
- **Streaming datasets**: Accepts datasets via WebSocket
- **Incremental learning**: Trains on each dataset immediately
- **Progress tracking**: Real-time training progress updates

### ✅ Validation & Safety
- **Dataset validation**: Ensures 1559 features + dual outcomes
- **Error handling**: Graceful error recovery
- **Memory management**: Automatic tensor cleanup

## WebSocket API

### Client → Server Messages

#### Train on Dataset
```json
{
  "type": "train",
  "dataset": [
    {
      "features": [/* 1559 feature values */],
      "outcomes": {
        "buy_outcome": 1.0,
        "sell_outcome": -1.0
      }
    }
  ]
}
```

#### Make Prediction
```json
{
  "type": "predict",
  "features": [/* 1559 feature values */]
}
```

#### Save Model
```json
{
  "type": "save"
}
```

#### Get Statistics
```json
{
  "type": "stats"
}
```

### Server → Client Messages

#### Training Progress
```json
{
  "type": "training_progress",
  "epoch": 1,
  "loss": 0.001234,
  "mae": 0.000567
}
```

#### Training Completed
```json
{
  "type": "training_completed",
  "samples": 100,
  "loss": 0.001234,
  "duration": 1500,
  "totalSamples": 50000
}
```

#### Prediction Result
```json
{
  "type": "prediction",
  "buy_outcome": 0.75,
  "sell_outcome": -0.45,
  "decision": "BUY",
  "confidence": 1.2
}
```

## Dataset Format

Each training sample must contain:

```json
{
  "features": [/* Array of exactly 1559 numbers */],
  "outcomes": {
    "buy_outcome": 1.0,   // +1.0=hit target, -1.0=hit stop, 0.0=no target
    "sell_outcome": -1.0  // +1.0=hit target, -1.0=hit stop, 0.0=no target
  }
}
```

## Usage Examples

### Basic Training Client
```javascript
const TrainingClient = require('./src/training-client')

const client = new TrainingClient()
await client.connect()

// Send training data
await client.sendTrainingData([
  {
    features: new Array(1559).fill(0), // Your market features
    outcomes: { buy_outcome: 1.0, sell_outcome: -1.0 }
  }
])

// Make prediction
await client.makePrediction(new Array(1559).fill(0.5))

// Save model
await client.saveModel()
```

### Data Generator Integration
```javascript
// Your data generator connects and streams datasets
const client = new TrainingClient()
await client.connect()

while (hasMoreData) {
  const dataset = await generateDataset() // Your data generation
  await client.sendTrainingData(dataset)
  await sleep(1000) // Rate limiting
}
```

## Configuration

Server configuration in `src/training-server.js`:

```javascript
this.config = {
  learningRate: 0.001,     // Adam optimizer learning rate
  batchSize: 32,           // Training batch size
  epochs: 1,               // Epochs per dataset (streaming)
  modelSavePath: './models/trading-model.json'
}
```

## Model Architecture

- **Input**: 1559 features (market data + indicators + patterns)
- **Hidden**: 6 dense layers with dropout (512→256→128→64→32)
- **Output**: 2 values (buy_outcome, sell_outcome)
- **Total**: ~973K parameters

## Performance Tips

### For High-Frequency Training
- Use smaller batch sizes (16-32)
- Single epoch per dataset
- Regular model saving (every N samples)

### For Large Datasets
- Increase batch size (64-128)
- Multiple epochs if needed
- Memory monitoring

## Monitoring

The server tracks:
- **Total samples trained**
- **Last training loss**
- **Training start time**
- **Last update timestamp**

Access via WebSocket `stats` message or server logs.

## Production Deployment

### Docker Ready
```dockerfile
FROM node:18
COPY . /app
WORKDIR /app
RUN npm install
EXPOSE 8080
CMD ["npm", "run", "training-server"]
```

### Security Considerations
- Add authentication for production
- Rate limiting for client connections
- Input validation and sanitization
- Secure model storage

## Next Steps

1. **Connect your data generator** to the WebSocket client
2. **Stream real market data** for training
3. **Monitor training progress** through WebSocket messages
4. **Scale horizontally** with multiple training servers

Ready to start training with real data! 🚀