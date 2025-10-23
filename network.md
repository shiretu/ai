# Trading Neural Network Specification

## 🎯 Overview
A deep learning model for cryptocurrency trading decisions based on BTC/USDC historical data from Binance. The network analyzes 2-hour windows of trading data to predict optimal buy/sell decisions.

## 📊 Input Specification

### Core Architecture
- **Time Window**: 120 consecutive 1-minute candles (2 hours)
- **Prediction Target**: Binary trading decision with confidence
- **Validation Window**: Following 60 minutes of actual trades (1 hour)

### Input Structure (Total: 1,559 features)

#### 1. Candle Data (120 × 8 = 960 features)
For each of the 120 candles:
```
- open: Opening price
- high: Highest price in period  
- low: Lowest price in period
- close: Closing price
- baseVolume: Volume in BTC
- quoteVolume: Volume in USDC
- timestamp: Unix timestamp
- candleColor: +1 (green), -1 (red), 0 (doji)
```

#### 2. Technical Studies (120 × 3 = 360 features)
For each of the 120 candles:
```
- EMA12: 12-period Exponential Moving Average
- EMA26: 26-period Exponential Moving Average
- Signal9: 9-period EMA of MACD line (EMA12 - EMA26)
```

#### 3. Pattern Recognition (237 features)
Sliding window pattern detection:
```
- 2-candle patterns: 119 patterns (positions 1-2 through 119-120)
- 3-candle patterns: 118 patterns (positions 1-3 through 118-120)
```
Each pattern encoded as integer (0=none, 1=hammer, 2=doji, 3=engulfing, etc.)

#### 4. Global Context (2 features)
```
- candleDuration: Duration in minutes (typically 1)
- windowSize: Number of candles in sequence (120)
```

### Mathematical Formula
```
Total Input Size = N(M + 8) + (2N - 3) + 2
Where:
- N = 120 (number of candles)
- M = 3 (number of studies per candle)
- 8 = OHLC + volumes + timestamp + color
- (2N - 3) = 237 (pattern features)
- 2 = global context (duration + window size)

Result: 120(3 + 8) + 237 + 2 = 1,559 features
```

## 🎯 Output Specification

### Network Output
- **Single continuous value**: Range [-1, +1]
- **Interpretation**:
  - `+1.0`: Strong BUY signal
  - `0.0`: Neutral/uncertain (no trade recommended)
  - `-1.0`: Strong SELL signal
  - Intermediate values indicate confidence level

### Trading Thresholds (suggested)
```
|output| > 0.7: High confidence trade
|output| > 0.3: Medium confidence trade  
|output| ≤ 0.3: No trade (too uncertain)
```

## 📈 Training Data Structure

### Sample Generation
1. **Random sampling** from 4+ years of BTC/USDC data
2. **Each sample contains**:
   - 120 consecutive candles (input features)
   - Next 60 minutes of actual trades (validation data)

### Ground Truth Generation
- Execute network's trading decision on actual subsequent trades
- Measure percentage gain/loss over the following hour
- **Training target**: Actual % return (normalized to [-1, +1] range)

### Training Process
```
Input: 1,558 features → Network → Output: [-1, +1]
Validation: Simulate trade on next hour → Actual % return
Loss: Mean Squared Error between predicted signal and actual return
```

## 🧠 Network Architecture Considerations

### Input Layer
- **Size**: 1,559 neurons
- **Normalization**: Essential due to different feature scales
  - Price data: Normalized by recent price range
  - Volume data: Log-normalized
  - Studies: Already normalized (EMAs)
  - Patterns: One-hot or ordinal encoding

### Hidden Layers (suggested)
- **Dense layers** with dropout for regularization
- **Possible LSTM/GRU** components for temporal patterns
- **Attention mechanisms** to focus on relevant time periods

### Output Layer
- **Single neuron** with tanh activation (outputs [-1, +1])

## 📋 Data Pipeline

### Preprocessing Steps
1. **Candle generation** from raw trade data
2. **Technical indicator calculation** (EMA12, EMA26, Signal9)
3. **Pattern detection** (2 and 3-candle patterns)
4. **Feature normalization** and scaling
5. **Sample generation** with random windows

### Training Data Format
```json
{
  "input": [1559 normalized features],
  "target": 0.75,
  "validation_trades": [
    {"price": 45230.50, "qty": 0.1, "timestamp": 1698123456},
    ...
  ],
  "actual_return": 0.023
}
```

## 🎯 Success Metrics

### Primary Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst losing streak
- **Win Rate**: Percentage of profitable trades
- **Average Return per Trade**: Mean profit/loss

### Secondary Metrics
- **Prediction Accuracy**: How often direction is correct
- **Calibration**: Do confidence levels match actual performance
- **Volatility**: Standard deviation of returns

## 💡 Implementation Notes

### Computational Requirements
- **Input size**: 1,559 features per sample
- **GPU acceleration**: Recommended for training
- **Memory**: Substantial for large datasets (4+ years of data)

### Real-time Inference
- **Latency target**: < 100ms for live trading
- **Feature pipeline**: Real-time calculation of studies and patterns
- **Model serving**: Optimized inference engine

---

**Total Feature Count**: 1,559 inputs → 1 output
**Training Validation**: Real trading simulation on subsequent hour
**Objective**: Maximize risk-adjusted returns while minimizing drawdown