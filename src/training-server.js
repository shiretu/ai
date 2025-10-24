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
                    inputShape: [1559],
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

                // Output layer - 2 outputs for buy_outcome and sell_outcome
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

            if (!sample.features || !Array.isArray(sample.features)) {
                throw new Error(`Sample ${i}: features must be an array`)
            }

            if (sample.features.length !== 1559) {
                throw new Error(`Sample ${i}: features must have exactly 1559 values, got ${sample.features.length}`)
            }

            if (!sample.outcomes || typeof sample.outcomes !== 'object') {
                throw new Error(`Sample ${i}: outcomes must be an object`)
            }

            if (typeof sample.outcomes.buy !== 'number' || typeof sample.outcomes.sell !== 'number') {
                throw new Error(`Sample ${i}: buy and sell must be numbers`)
            }
        }
    }

    /**
     * Prepare training data tensors
     */
    prepareTrainingData (dataset) {
        const features = []
        const labels = []

        for (const sample of dataset) {
            features.push(sample.features)
            labels.push([sample.outcomes.buy, sample.outcomes.sell])
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
        if (!features || features.length !== 1559) {
            throw new Error('Features must be an array of exactly 1559 values')
        }

        const input = tf.tensor2d([features])
        const prediction = this.model.predict(input)
        const result = await prediction.data()

        ws.send(JSON.stringify({
            type: 'prediction',
            buy: result[0],
            sell: result[1],
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
