# AI Trading Project - Context Dump
**Date:** October 24, 2025  
**Status:** Training Data Structure Complete

## Project Overview

Started as "prepare me a very very very simple nodejs project called ai" and evolved into a sophisticated BTC/USDC trading neural network with comprehensive architecture and training data structure.

## Current Project State

### Core Architecture
- **Node.js project** with TensorFlow.js
- **1,559 input features** for trading analysis
- **2 output predictions** (buy_outcome, sell_outcome)
- **Clean project structure** with single working implementation

### Trading Strategy
- **Simultaneous dual evaluation**: Network evaluates both BUY and SELL scenarios independently
- **Profit target**: +0.7% (0.5% net after 0.2% fees)
- **Stop loss**: -0.4% (0.2% net after 0.2% fees)
- **Output values**: +1.0 (hit target), -1.0 (hit stop), 0.0 (no target hit)

### Decision Logic
1. Network predicts: `[buy_confidence, sell_confidence]`
2. System chooses action with highest confidence
3. Execute chosen action
4. Provide feedback on actual outcome
5. Network learns from both chosen and alternative outcomes

## File Structure

```
/Users/shiretu/work/ai/
├── src/
│   ├── trading-network.js     # Main implementation (reads JSON, builds TF.js model)
│   ├── llm.js                # Original LLM sequence prediction
│   ├── test.js               # Testing utilities
│   └── check-backend.js      # TensorFlow.js backend validation
├── network_architecture.json # Language-agnostic neural network specification
├── training_data.json        # Training data with dual outcomes structure
├── package.json              # Clean dependencies (just @tensorflow/tfjs)
├── python/                   # GPU training utilities
└── README.md
```

## Key Files Details

### network_architecture.json
- **Complete model specification** with 1,559 input features
- **Multi-branch architecture**: temporal, pattern, global processing
- **973K parameters** total
- **Language-agnostic design** for cross-platform implementation

### training_data.json (Current Structure)
```json
{
  "metadata": {
    "feature_count": 1559,
    "output_structure": {
      "buy_outcome": "BUY order outcome: +1.0 (target), -1.0 (stop), 0.0 (no hit)",
      "sell_outcome": "SELL order outcome: +1.0 (target), -1.0 (stop), 0.0 (no hit)"
    }
  },
  "training_data": [
    {
      "features": "an array of 1559 normalized feature values",
      "outcomes": {
        "buy_outcome": 1.0,
        "sell_outcome": -1.0
      }
    }
  ]
}
```

### src/trading-network.js
- **Working implementation** that reads network_architecture.json
- **Builds TensorFlow.js model** with 973K parameters
- **6-layer dense network** with dropout
- **Ready for training** with real data

## Feature Breakdown (1,559 total)

1. **Candle Data (0-959)**: 120 consecutive 1-min candles × 8 features
   - OHLC, volume, timestamp, color, body_size
2. **Studies Data (960-1319)**: 120 candles × 3 technical indicators  
   - SMA, EMA, RSI for each candle
3. **Pattern Data (1320-1556)**: 237 candlestick patterns
   - 119 single patterns + 118 sliding window patterns
4. **Global Context (1557-1558)**: 2 global parameters
   - Duration (1min), window size (120)

## Technical Progress

### Completed ✅
- ✅ Project structure and dependencies
- ✅ Neural network architecture design
- ✅ Language-agnostic JSON specification
- ✅ Working Node.js implementation
- ✅ Training data structure with dual outcomes
- ✅ File cleanup and organization
- ✅ Trading strategy definition

### Ready for Implementation 🚀
- **Real data population**: Replace "an array of 1559 normalized feature values" with actual market data
- **Training execution**: Use training_data.json to train the network
- **Backtesting**: Validate with historical data
- **Live deployment**: Real-time trading implementation

## Key Insights from Development

### Network Design Evolution
- Started with single output (buy/sell/hold signal)
- **Evolved to dual outputs** for independent BUY/SELL evaluation
- This allows complete "what if?" analysis for both scenarios

### Training Philosophy
- **Supervised learning**: Network needs both outcomes to learn properly
- **Comparative learning**: Network learns to choose better option by seeing both alternatives
- **Risk assessment**: Learns when both directions are risky (both negative)

### Trading Logic
- **Independent evaluation**: Both BUY and SELL orders evaluated to completion
- **Decision by confidence**: Choose action with highest predicted outcome
- **No-trade option**: When both predictions are negative, don't trade

## TensorFlow.js Compatibility
- **Resolved macOS issues** by using basic @tensorflow/tfjs package
- **Removed @tensorflow/tfjs-node** dependency for better compatibility
- **973K parameters** building successfully

## Next Steps Priority

1. **Data Collection**: Gather real BTC/USDC 1-minute data
2. **Feature Engineering**: Calculate technical indicators and patterns
3. **Data Preprocessing**: Normalize features to 0-1 range
4. **Training Pipeline**: Implement batch training with validation
5. **Backtesting Framework**: Historical performance validation
6. **Live Trading Interface**: Real-time execution system

## Important Context for Resumption

- **Network expects 1,559 features exactly**
- **Training data must have both buy_outcome AND sell_outcome**
- **All features should be normalized**
- **Current implementation is CPU-only (TensorFlow.js basic)**
- **Python GPU training pipeline also available in python/ directory**

## Development Philosophy

The project evolved from simple experimentation to production-ready trading system with:
- **Comprehensive feature engineering** (120 candles + studies + patterns)
- **Robust architecture** (multi-branch neural network)
- **Practical trading strategy** (dual order evaluation with clear risk management)
- **Clean implementation** (single working file, no redundancy)

## Status Summary
**READY FOR REAL DATA** - All infrastructure complete, need market data to populate training_data.json features arrays and begin training.