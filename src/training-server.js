/**
 * Real-time Trading Network Training Server
 * Accepts training datasets via WebSocket and performs incremental training
 */

const WebSocket = require('ws')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node') // Enable Node.js backend for file operations
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
            modelName: 'myFirstModel'
        }

        this.modelRootPath = path.resolve(path.join(__dirname, '..', 'models', this.config.modelName))
        this.architecturePath = path.join(this.modelRootPath, 'architecture.json')
        this.tfModelFolder = path.join(this.modelRootPath, 'tf')
        this.tfModelPath = path.join(this.tfModelFolder, 'model.json')
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
            // Load architecture configuration
            await this.loadArchitecture()

            // Try to load existing model
            // tf.loadLayersModel() automatically loads both model.json and weights.bin
            // from the directory path when given the model.json file path
            if (await this.modelExists()) {
                console.log('📁 Loading existing model from disk...')
                console.log(`   Model: ${this.tfModelPath}`)
                this.model = await tf.loadLayersModel(`file://${this.tfModelPath}`)
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
     * Load network architecture from JSON file
     */
    async loadArchitecture () {
        try {
            this.architecture = JSON.parse(await fs.readFile(this.architecturePath, 'utf8'))
            console.log(`📐 Architecture loaded: ${this.architecture.input_features} → ${this.architecture.output_features}`)
        } catch (error) {
            console.error(`❌ FATAL: ${this.architecturePath} is required but could not be loaded`)
            console.error(`   Path: ${this.architecturePath}`)
            console.error(`   Error: ${error.message}`)
            throw new Error(`${this.architecturePath} is required - server cannot start without it`)
        }
    }

    /**
     * Check if model files exist
     * TensorFlow.js saves/loads models as a pair: model.json + weights.bin
     * Both files must be present for successful model loading
     */
    async modelExists () {
        try {
            // Check both model.json and weights.bin files exist
            await fs.access(this.tfModelPath)
            return true
        } catch {
            return false
        }
    }

    /**
     * Create model from loaded architecture JSON
     */
    createModel () {
        console.log('🧠 Building neural network architecture...')
        const layerFactory = {
            dense: (config, isFirstLayer) => {
                const layerConfig = {
                    units: config.units,
                    activation: config.activation
                }

                // Add input shape for first layer
                if (isFirstLayer) {
                    layerConfig.inputShape = [this.architecture.input_features]
                }

                // Add name if specified
                if (config.name) {
                    layerConfig.name = config.name
                }

                return tf.layers.dense(layerConfig)
            },
            dropout: (config) => tf.layers.dropout({ rate: config.rate })
        }

        return tf.sequential({
            layers: this.architecture.layers.map((layerConfig, index) => {
                const factoryFunc = layerFactory[layerConfig.type]
                if (!factoryFunc) {
                    console.warn(`⚠️  Unknown layer type: ${layerConfig.type}, skipping`)
                    return null
                }
                return factoryFunc(layerConfig, index === 0)
            }).filter(layer => layer !== null)
        })
    }

    /**
     * Compile model for training
     */
    compileModel () {
        // Use optimizer from architecture or fallback to config
        let optimizer
        if (this.architecture && this.architecture.optimizer) {
            const optimizerConfig = this.architecture.optimizer
            switch (optimizerConfig.type) {
                case 'adam':
                    optimizer = tf.train.adam(optimizerConfig.learning_rate || 0.001)
                    break
                default:
                    optimizer = tf.train.adam(this.config.learningRate)
            }
        } else {
            optimizer = tf.train.adam(this.config.learningRate)
        }

        // Use loss and metrics from architecture or fallback to defaults
        const loss = this.architecture?.loss || 'meanSquaredError'
        const metrics = this.architecture?.metrics || ['mae']

        this.model.compile({
            optimizer,
            loss,
            metrics
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
                    await this.handleRequest(JSON.parse(data.toString()), (error, result) => {
                        if (error) {
                            ws.send(JSON.stringify({
                                type: 'error',
                                message: error.message
                            }))
                        } else {
                            ws.send(JSON.stringify(result))
                        }
                    })
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
     * Handle incoming WebSocket requests
     */
    async handleRequest (message, callback) {
        try {
            switch (message.type) {
                case 'train':
                    await this.trainOnSample(message.sample, callback)
                    break

                case 'save':
                    callback(null, await this.saveModel())
                    break

                case 'predict':
                    callback(null, await this.makePrediction(message.features))
                    break

                case 'stats':
                    callback(null, this.getStats())
                    break

                default:
                    callback(new Error(`Unknown message type: ${message.type}`))
            }
        } catch (error) {
            callback(error)
        }
    }

    /**
     * Train on a single sample
     */
    async trainOnSample (sample, callback) {
        console.log('🎯 Training on single sample')

        // Prepare training data
        const { features, labels } = this.prepareSampleData(sample)

        // Start training
        const startTime = Date.now()

        callback(null, {
            type: 'training_started',
            samples: 1,
            timestamp: new Date().toISOString()
        })

        try {
            // Train the model
            const history = await this.model.fit(features, labels, {
                epochs: this.config.epochs,
                batchSize: 1,
                verbose: 0,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        callback(null, {
                            type: 'training_progress',
                            epoch: epoch + 1,
                            loss: logs.loss,
                            mae: logs.mae
                        })
                    }
                }
            })

            // Update stats
            this.updateTrainingStats(1, history.history.loss[0])

            // Training completed
            const duration = Date.now() - startTime

            callback(null, {
                type: 'training_completed',
                samples: 1,
                loss: history.history.loss[0],
                mae: history.history.mae[0],
                duration,
                totalSamples: this.trainingStats.totalSamples
            })
        } catch (error) {
            callback(error)
        } finally {
            // Clean up tensors
            features.dispose()
            labels.dispose()
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
     * Prepare single sample data tensors
     */
    prepareSampleData (sample) {
        const flatFeatures = this.flattenFeatures(sample.features)
        const labels = [sample.outcomes.grossBuy, sample.outcomes.grossSell]

        return {
            features: tf.tensor2d([flatFeatures]),
            labels: tf.tensor2d([labels])
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
     * TensorFlow.js model.save() creates two files in the target directory:
     * - model.json: Contains the model architecture and metadata
     * - weights.bin: Contains the trained model weights
     */
    async saveModel (ws) {
        try {
            console.log('💾 Saving model to disk...')
            console.log(`   Target: ${this.modelRootPath}/`)
            // model.save() will create both model.json and weights.bin in the directory
            await fs.mkdir(this.tfModelFolder, { recursive: true })
            await this.model.save(`file://${this.tfModelFolder}`)
            console.log('✅ Model saved successfully')
            console.log(`   Created: ${this.tfModelPath}`)

            return {
                type: 'model_saved',
                path: this.modelRootPath,
                timestamp: new Date().toISOString()
            }
        } catch (error) {
            throw new Error(`Failed to save model: ${error.message}`)
        }
    }

    /**
     * Make prediction
     */
    async makePrediction (features) {
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

        const ret = {
            type: 'prediction',
            grossBuy: result[0],
            grossSell: result[1],
            decision: result[0] > result[1] ? 'BUY' : (result[1] > result[0] ? 'SELL' : 'HOLD'),
            confidence: Math.abs(result[0] - result[1])
        }

        // Clean up
        input.dispose()
        prediction.dispose()

        return ret
    }

    /**
     * Send training statistics
     */
    getStats (ws) {
        return {
            type: 'stats',
            stats: this.trainingStats,
            modelParams: this.model.countParams(),
            config: this.config
        }
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
