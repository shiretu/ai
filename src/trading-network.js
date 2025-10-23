const tf = require('@tensorflow/tfjs')
const fs = require('fs')

/**
 * Simple Trading Network Builder
 * Uses only @tensorflow/tfjs (no node backend)
 */
async function buildTradingNetworkSimple () {
    console.log('🚀 Trading Neural Network - Simple Version')
    console.log('==========================================\n')

    // Load specification
    console.log('📋 Loading model specification...')
    const specData = fs.readFileSync('./network_architecture.json', 'utf8')
    const spec = JSON.parse(specData).model_specification

    console.log(`✅ Loaded: ${spec.name}`)
    console.log(`📊 Input features: ${spec.input_schema.total_features}`)
    console.log(`🎯 Output range: [${spec.output_schema.range.join(', ')}]`)

    // Build model architecture from JSON spec
    console.log('\n🏗️  Building neural network...')

    // Create input layer
    const input = tf.input({
        shape: [spec.input_schema.total_features],
        name: 'market_data'
    })

    // Sequential processing based on spec
    let layer = tf.layers.dense({
        units: 512,
        activation: 'relu',
        name: 'temporal_processing'
    }).apply(input)

    layer = tf.layers.dropout({ rate: 0.2 }).apply(layer)

    layer = tf.layers.dense({
        units: 256,
        activation: 'relu',
        name: 'pattern_processing'
    }).apply(layer)

    layer = tf.layers.dropout({ rate: 0.3 }).apply(layer)

    layer = tf.layers.dense({
        units: 128,
        activation: 'relu',
        name: 'feature_fusion'
    }).apply(layer)

    layer = tf.layers.dropout({ rate: 0.3 }).apply(layer)

    layer = tf.layers.dense({
        units: 64,
        activation: 'relu',
        name: 'decision_processing'
    }).apply(layer)

    layer = tf.layers.dropout({ rate: 0.2 }).apply(layer)

    layer = tf.layers.dense({
        units: 32,
        activation: 'relu',
        name: 'final_processing'
    }).apply(layer)

    // Output layer: trading signal [-1, +1]
    const output = tf.layers.dense({
        units: 1,
        activation: 'tanh',
        name: 'trading_signal'
    }).apply(layer)

    // Create model
    const model = tf.model({
        inputs: input,
        outputs: output,
        name: spec.name
    })

    console.log('✅ Model architecture built!')

    // Compile with spec configuration
    const trainingConfig = spec.training_config
    console.log('⚙️  Compiling model...')

    model.compile({
        optimizer: tf.train.adam(trainingConfig.optimizer.learning_rate),
        loss: 'meanSquaredError',
        metrics: ['mae']
    })

    console.log('✅ Model compiled!')

    // Show model summary
    console.log('\n📊 MODEL SUMMARY')
    console.log('================')
    model.summary()

    console.log('\n📋 Model Statistics:')
    console.log(`   Total parameters: ${model.countParams().toLocaleString()}`)
    console.log(`   Input features: ${spec.input_schema.total_features}`)
    console.log(`   Expected in spec: ${spec.model_metadata.total_parameters}`)

    // Test the model with sample data
    console.log('\n🧪 TESTING MODEL')
    console.log('================')

    // Create test data
    const batchSize = 3
    console.log(`Creating test data: [${batchSize}, ${spec.input_schema.total_features}]`)

    // Simple test data (random values)
    const testData = []
    for (let i = 0; i < batchSize; i++) {
        const sample = []
        for (let j = 0; j < spec.input_schema.total_features; j++) {
            sample.push(Math.random() * 2 - 1) // Random values between -1 and 1
        }
        testData.push(sample)
    }

    const testTensor = tf.tensor2d(testData)
    console.log(`Input tensor shape: [${testTensor.shape.join(', ')}]`)

    // Make predictions
    const predictions = model.predict(testTensor)
    console.log(`Output tensor shape: [${predictions.shape.join(', ')}]`)

    // Get prediction values
    const predictionData = await predictions.data()

    console.log('\n📈 Sample Predictions:')
    for (let i = 0; i < batchSize; i++) {
        const pred = predictionData[i]
        let signal

        if (pred > 0.7) signal = 'Strong BUY 🚀'
        else if (pred > 0.3) signal = 'Weak BUY 📈'
        else if (pred >= -0.3) signal = 'HOLD ⏸️'
        else if (pred >= -0.7) signal = 'Weak SELL 📉'
        else signal = 'Strong SELL 💥'

        console.log(`   Sample ${i + 1}: ${pred.toFixed(4)} → ${signal}`)
    }

    // Show feature breakdown from spec
    console.log('\n📝 INPUT FEATURE STRUCTURE')
    console.log('==========================')

    const groups = spec.input_schema.feature_groups
    Object.entries(groups).forEach(([name, info]) => {
        const count = info.end_index - info.start_index
        console.log(`${name.padEnd(15)}: indices ${info.start_index.toString().padStart(4)}-${(info.end_index - 1).toString().padStart(4)} (${count.toString().padStart(3)} features)`)
        console.log(`${' '.repeat(17)}${info.description}`)
    })

    // Show trading signal interpretation
    console.log('\n💹 TRADING SIGNAL GUIDE')
    console.log('=======================')
    Object.entries(spec.output_schema.interpretation).forEach(([value, meaning]) => {
        console.log(`${value.toString().padEnd(6)} → ${meaning}`)
    })

    // Training info
    console.log('\n⚙️  READY FOR TRAINING')
    console.log('=====================')
    console.log(`Optimizer: ${trainingConfig.optimizer.type}`)
    console.log(`Learning rate: ${trainingConfig.optimizer.learning_rate}`)
    console.log(`Batch size: ${trainingConfig.batch_size}`)
    console.log(`Epochs: ${trainingConfig.epochs}`)

    console.log('\n🎯 NEXT STEPS')
    console.log('=============')
    console.log('1. Prepare BTC/USDC trading data in 1,559 feature format')
    console.log('2. Train model with real historical data')
    console.log('3. Validate with trading simulations')
    console.log('4. Deploy for live trading')

    // Clean up tensors
    testTensor.dispose()
    predictions.dispose()

    console.log('\n🎉 Trading network successfully built from JSON specification!')
    console.log('✅ Model is ready for training with real trading data')

    return model
}

// Run demo
if (require.main === module) {
    buildTradingNetworkSimple().catch(error => {
        console.error('❌ Error:', error.message)
        console.error(error.stack)
        process.exit(1)
    })
}

module.exports = { buildTradingNetworkSimple }
