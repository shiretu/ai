/**
 * Real-time Trading Network Training Server
 * Accepts training datasets via WebSocket and performs incremental training
 */

const WebSocket = require('ws')
const tf = require('@tensorflow/tfjs')
const fs = require('fs').promises
const path = require('path')

class TradingTrainingServer {
    constructor (port = 8080) {
        this.port = port
        this.model = null
        this.wss = null
        this.trainingStats = {
            totalSamples: 0,
            lastLoss: null,
            trainingStarted: null,
            lastUpdate: null
        }

        // Training configuration
        this.config = {
            learningRate: 0.001,
            batchSize: 32,
            epochs: 1, // Single epoch per dataset for real-time training
            validationSplit: 0.0, // No validation split for streaming
            modelSavePath: './models/trading-model.json'
        }
    }

    /**
     * Initialize the training server
     */
    async initialize () {
        console.log('🚀 Initializing Trading Training Server...')

        // Load or create model
        await this.loadOrCreateModel()

        // Start WebSocket server
        this.startWebSocketServer()

        console.log(`✅ Training server running on port ${this.port}`)
        console.log(`📊 Model loaded with ${this.model.countParams()} parameters`)
        console.log('🔗 Waiting for WebSocket connections...')
    }

    /**
     * Load existing model or create new one
     */
    async loadOrCreateModel () {
        try {
            // Try to load existing model
            if (await this.modelExists()) {
                console.log('📁 Loading existing model from disk...')
                this.model = await tf.loadLayersModel(`file://${path.resolve(this.config.modelSavePath)}`)
                console.log('✅ Existing model loaded successfully')
            } else {
                console.log('🏗️  Creating new model...')
                this.model = this.createModel()
                console.log('✅ New model created successfully')
            }

            // Compile model for training
            this.compileModel()
        } catch (error) {
            console.error('❌ Error loading model:', error)
            console.log('🏗️  Creating fallback model...')
            this.model = this.createModel()
            this.compileModel()
        }
    }

    /**
     * Check if model file exists
     */
    async modelExists () {
        try {
            await fs.access(this.config.modelSavePath)
            return true
        } catch {
            return false
        }
    }

    /**
     * Create new neural network model
     */
    createModel () {
        console.log('🧠 Building neural network architecture...')

        const model = tf.sequential({
            layers: [
                // Input layer
                tf.layers.dense({
                    inputShape: [2043],
                    units: 512,
                    activation: 'relu',
                    name: 'temporal_processing'
                }),
                tf.layers.dropout({ rate: 0.3 }),

                // Hidden layers
                tf.layers.dense({
                    units: 256,
                    activation: 'relu',
                    name: 'pattern_processing'
                }),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({
                    units: 128,
                    activation: 'relu',
                    name: 'feature_fusion'
                }),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({
                    units: 64,
                    activation: 'relu',
                    name: 'decision_processing'
                }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({
                    units: 32,
                    activation: 'relu',
                    name: 'final_processing'
                }),

                // Output layer - 2 outputs for grossBuy and grossSell
                tf.layers.dense({
                    units: 2,
                    activation: 'linear', // Linear for regression (-1 to +1 range)
                    name: 'trading_outcomes'
                })
            ]
        })

        return model
    }

    /**
     * Compile model for training
     */
    compileModel () {
        this.model.compile({
            optimizer: tf.train.adam(this.config.learningRate),
            loss: 'meanSquaredError',
            metrics: ['mae']
        })
    }

    /**
     * Start WebSocket server
     */
    startWebSocketServer () {
        this.wss = new WebSocket.Server({ port: this.port })

        this.wss.on('connection', (ws) => {
            console.log('🔗 New WebSocket connection established')

            ws.on('message', async (data) => {
                try {
                    await this.handleTrainingData(ws, data)
                } catch (error) {
                    console.error('❌ Error processing training data:', error)
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: error.message
                    }))
                }
            })

            ws.on('close', () => {
                console.log('🔌 WebSocket connection closed')
            })

            // Send welcome message
            ws.send(JSON.stringify({
                type: 'connected',
                message: 'Training server ready',
                modelParams: this.model.countParams(),
                trainingStats: this.trainingStats
            }))
        })
    }

    /**
     * Handle incoming training data
     */
    async handleTrainingData (ws, data) {
        const message = JSON.parse(data.toString())

        switch (message.type) {
            case 'train':
                await this.trainOnDataset(ws, message.dataset)
                break

            case 'save':
                await this.saveModel(ws)
                break

            case 'predict':
                await this.makePrediction(ws, message.features)
                break

            case 'stats':
                this.sendStats(ws)
                break

            default:
                ws.send(JSON.stringify({
                    type: 'error',
                    message: `Unknown message type: ${message.type}`
                }))
        }
    }

    /**
     * Train on a single dataset
     */
    async trainOnDataset (ws, dataset) {
        console.log(`🎯 Training on dataset with ${dataset.length} samples`)

        if (!dataset || dataset.length === 0) {
            throw new Error('Dataset is empty or invalid')
        }

        // Validate dataset format
        this.validateDataset(dataset)

        // Prepare training data
        const { features, labels } = this.prepareTrainingData(dataset)

        // Start training
        const startTime = Date.now()

        ws.send(JSON.stringify({
            type: 'training_started',
            samples: dataset.length,
            timestamp: new Date().toISOString()
        }))

        try {
            // Train the model
            const history = await this.model.fit(features, labels, {
                epochs: this.config.epochs,
                batchSize: Math.min(this.config.batchSize, dataset.length),
                verbose: 0,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        ws.send(JSON.stringify({
                            type: 'training_progress',
                            epoch: epoch + 1,
                            loss: logs.loss,
                            mae: logs.mae
                        }))
                    }
                }
            })

            // Update stats
            this.updateTrainingStats(dataset.length, history.history.loss[0])

            // Training completed
            const duration = Date.now() - startTime

            ws.send(JSON.stringify({
                type: 'training_completed',
                samples: dataset.length,
                loss: history.history.loss[0],
                mae: history.history.mae[0],
                duration,
                totalSamples: this.trainingStats.totalSamples
            }))

            console.log(`✅ Training completed in ${duration}ms, Loss: ${history.history.loss[0].toFixed(6)}`)
        } finally {
            // Clean up tensors
            features.dispose()
            labels.dispose()
        }
    }

    /**
     * Validate dataset format
     */
    validateDataset (dataset) {
        for (let i = 0; i < dataset.length; i++) {
            const sample = dataset[i]

            if (!sample.features || typeof sample.features !== 'object') {
                throw new Error(`Sample ${i}: features must be an object`)
            }

            // Validate nested feature structure
            const flatFeatures = this.flattenFeatures(sample.features)
            if (flatFeatures.length !== 2043) {
                throw new Error(`Sample ${i}: features must have exactly 2043 values, got ${flatFeatures.length}`)
            }

            if (!sample.outcomes || typeof sample.outcomes !== 'object') {
                throw new Error(`Sample ${i}: outcomes must be an object`)
            }

            if (typeof sample.outcomes.grossBuy !== 'number' || typeof sample.outcomes.grossSell !== 'number') {
                throw new Error(`Sample ${i}: grossBuy and grossSell must be numbers`)
            }
        }
    }

    /**
     * Flatten nested features into a single array
     */
    flattenFeatures (features) {
        return [
            // Candles: 8 arrays × 120 = 960 features
            ...features.candles.opens,
            ...features.candles.highs,
            ...features.candles.lows,
            ...features.candles.closes,
            ...features.candles.volumes,
            ...features.candles.timestamps,
            ...features.candles.colors,
            ...features.candles.bodySizes,

            // Studies: 7 arrays × 120 = 840 features
            ...features.studies.sma9,
            ...features.studies.sma12,
            ...features.studies.sma21,
            ...features.studies.ema9,
            ...features.studies.ema12,
            ...features.studies.ema21,
            ...features.studies.rsi14,

            // Patterns: 237 features
            ...features.patterns.single,
            ...features.patterns.sliding,

            // Global: 6 features
            features.global.candleDuration,
            features.global.windowSize,
            features.global.grossProfitTarget,
            features.global.grossStopLoss,
            features.global.positionSize,
            features.global.fees
        ]
    }

    /**
     * Prepare training data tensors
     */
    prepareTrainingData (dataset) {
        const features = []
        const labels = []

        for (const sample of dataset) {
            features.push(this.flattenFeatures(sample.features))
            labels.push([sample.outcomes.grossBuy, sample.outcomes.grossSell])
        }

        return {
            features: tf.tensor2d(features),
            labels: tf.tensor2d(labels)
        }
    }

    /**
     * Update training statistics
     */
    updateTrainingStats (samples, loss) {
        this.trainingStats.totalSamples += samples
        this.trainingStats.lastLoss = loss
        this.trainingStats.lastUpdate = new Date().toISOString()

        if (!this.trainingStats.trainingStarted) {
            this.trainingStats.trainingStarted = new Date().toISOString()
        }
    }

    /**
     * Save model to disk
     */
    async saveModel (ws) {
        try {
            console.log('💾 Saving model to disk...')
            await this.model.save(`file://${path.resolve(this.config.modelSavePath)}`)

            ws.send(JSON.stringify({
                type: 'model_saved',
                path: this.config.modelSavePath,
                timestamp: new Date().toISOString()
            }))

            console.log('✅ Model saved successfully')
        } catch (error) {
            throw new Error(`Failed to save model: ${error.message}`)
        }
    }

    /**
     * Make prediction
     */
    async makePrediction (ws, features) {
        // Handle both nested object and flat array formats
        let flatFeatures
        if (Array.isArray(features)) {
            if (features.length !== 2043) {
                throw new Error('Features array must have exactly 2043 values')
            }
            flatFeatures = features
        } else if (typeof features === 'object') {
            flatFeatures = this.flattenFeatures(features)
            if (flatFeatures.length !== 2043) {
                throw new Error('Features object must flatten to exactly 2043 values')
            }
        } else {
            throw new Error('Features must be an array or nested object')
        }

        const input = tf.tensor2d([flatFeatures])
        const prediction = this.model.predict(input)
        const result = await prediction.data()

        ws.send(JSON.stringify({
            type: 'prediction',
            grossBuy: result[0],
            grossSell: result[1],
            decision: result[0] > result[1] ? 'BUY' : (result[1] > result[0] ? 'SELL' : 'HOLD'),
            confidence: Math.abs(result[0] - result[1])
        }))

        // Clean up
        input.dispose()
        prediction.dispose()
    }

    /**
     * Send training statistics
     */
    sendStats (ws) {
        ws.send(JSON.stringify({
            type: 'stats',
            stats: this.trainingStats,
            modelParams: this.model.countParams(),
            config: this.config
        }))
    }
}

// Start the server
async function startServer () {
    const server = new TradingTrainingServer(8080)
    await server.initialize()
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down training server...')
    process.exit(0)
})

// Export for testing
module.exports = TradingTrainingServer

// Start server if run directly
if (require.main === module) {
    startServer().catch(console.error)
}
