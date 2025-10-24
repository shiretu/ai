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
    #port = 8080 /** @type {number} */
    #model = null /** @type {tf.LayersModel} */
    #wss = null /** @type {WebSocket.Server} */
    #trainingStats = {
        totalSamples: 0,
        lastLoss: null,
        trainingStarted: null,
        lastUpdate: null
    } /** @type {{totalSamples:number,lastLoss:number|null,trainingStarted:Date|null,lastUpdate:Date|null}} */

    #config = {
        learningRate: 0.001,
        batchSize: 32,
        epochs: 1, // Single epoch per dataset for real-time training
        validationSplit: 0.0, // No validation split for streaming
        modelName: 'myFirstModel'
    } /** @type {{learningRate:number,batchSize:number,epochs:number,validationSplit:number,modelName:string}} */

    #architecture = null /** @type {any} */
    #modelRootPath = path.resolve(path.join(__dirname, '..', 'models', this.#config.modelName)) /** @type {string} */
    #architecturePath = path.join(this.#modelRootPath, 'architecture.json') /** @type {string} */
    #tfModelFolder = path.join(this.#modelRootPath, 'tf') /** @type {string} */
    #tfModelPath = path.join(this.#tfModelFolder, 'model.json') /** @type {string} */

    /**
     * Create a new instance of the TradingTrainingServer
     * @returns {Promise<TradingTrainingServer>}
     */
    static async create () {
        const result = new TradingTrainingServer()
        await result.#initialize()
        return result
    }

    async #initialize () {
        await this.#loadOrCreateModel()
        this.#startWebSocketServer()
    }

    async #loadOrCreateModel () {
        try {
            this.#architecture = JSON.parse(await fs.readFile(this.#architecturePath, 'utf8'))
            this.#model = await tf.loadLayersModel(`file://${this.#tfModelPath}`)
        } catch (error) {
            this.#model = this.#createModel()
        }
        this.#compileModel()
        console.log(`Model with ${this.#model.countParams()} parameters activated`)
    }

    #createModel () {
        const layerFactory = {
            dense: (config, isFirstLayer) => {
                const layerConfig = {
                    units: config.units,
                    activation: config.activation
                }

                // Add input shape for first layer
                if (isFirstLayer) {
                    layerConfig.inputShape = [this.#architecture.input_features]
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
            layers: this.#architecture.layers.map((layerConfig, index) => {
                const factoryFunc = layerFactory[layerConfig.type]
                if (!factoryFunc) {
                    console.warn(`WARNING: Unknown layer type: ${layerConfig.type}, skipping`)
                    return null
                }
                return factoryFunc(layerConfig, index === 0)
            }).filter(layer => layer !== null)
        })
    }

    #compileModel () {
        const optimizerFactory = {
            adam: (learningRate) => tf.train.adam(learningRate)
        }
        this.#model.compile({
            optimizer: optimizerFactory[this.#architecture.optimizer.type](this.#architecture.optimizer.learning_rate),
            loss: this.#architecture.loss,
            metrics: this.#architecture.metrics
        })
    }

    #startWebSocketServer () {
        this.#wss = new WebSocket.Server({ port: this.#port })
        this.#wss.on('connection', (ws) => {
            ws.on('message', async (data) => {
                try {
                    await this.#handleRequest(JSON.parse(data.toString()), (error, result) => {
                        if (error) {
                            ws.send(JSON.stringify({ type: 'error', message: error.message }))
                        } else {
                            ws.send(JSON.stringify(result))
                        }
                    })
                } catch (error) {
                    ws.send(JSON.stringify({ type: 'error', message: error.message }))
                }
            })
            ws.on('close', () => { console.log('WebSocket connection closed') })
            ws.send(JSON.stringify({ type: 'connected', message: 'Training server ready', modelParams: this.#model.countParams(), trainingStats: this.#trainingStats }))
        })
        console.log(`Training server running on port ${this.#port}`)
    }

    async #handleRequest (message, callback) {
        try {
            switch (message.type) {
                case 'train':
                    await this.#trainOnSample(message.sample, callback)
                    break

                case 'save':
                    await this.#saveModel(callback)
                    break

                case 'predict':
                    await this.#makePrediction(message.features, callback)
                    break

                case 'stats':
                    callback(null, this.#getStats())
                    break

                default:
                    callback(new Error(`Unknown message type: ${message.type}`))
            }
        } catch (error) {
            callback(error)
        }
    }

    async #trainOnSample (sample, callback) {
        const { features, labels } = this.#prepareTensors(sample)
        const startTime = Date.now()

        try {
            // Train the model
            const history = await this.#model.fit(features, labels, {
                epochs: this.#config.epochs,
                batchSize: 1,
                verbose: 0
            })

            // Update stats
            this.#updateTrainingStats(1, history.history.loss[0])

            // Training completed
            const duration = Date.now() - startTime

            callback(null, {
                type: 'training_completed',
                samples: 1,
                loss: history.history.loss[0],
                mae: history.history.mae[0],
                duration,
                totalSamples: this.#trainingStats.totalSamples
            })
        } catch (error) {
            callback(error)
        } finally {
            features.dispose()
            labels.dispose()
        }
    }

    #makeFlat (features) {
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

    #prepareTensors (sample) {
        const flatFeatures = this.#makeFlat(sample.features)
        const labels = [sample.outcomes.grossBuy, sample.outcomes.grossSell]

        return {
            features: tf.tensor2d([flatFeatures]),
            labels: tf.tensor2d([labels])
        }
    }

    #updateTrainingStats (samples, loss) {
        this.#trainingStats.totalSamples += samples
        this.#trainingStats.lastLoss = loss
        this.#trainingStats.lastUpdate = new Date().toISOString()

        if (!this.#trainingStats.trainingStarted) {
            this.#trainingStats.trainingStarted = new Date().toISOString()
        }
    }

    async #saveModel (callback) {
        try {
            await fs.mkdir(this.#tfModelFolder, { recursive: true })
            await this.#model.save(`file://${this.#tfModelFolder}`)
            callback(null, { type: 'model_saved', path: this.#tfModelFolder })
        } catch (error) {
            callback(error)
        }
    }

    async #makePrediction (features, callback) {
        // Handle both nested object and flat array formats
        let flatFeatures
        if (Array.isArray(features)) {
            if (features.length !== 2043) {
                callback(new Error('Features array must have exactly 2043 values'))
                return
            }
            flatFeatures = features
        } else if (typeof features === 'object') {
            flatFeatures = this.#makeFlat(features)
            if (flatFeatures.length !== 2043) {
                callback(new Error('Features object must flatten to exactly 2043 values'))
                return
            }
        } else {
            callback(new Error('Features must be an array or nested object'))
            return
        }

        const input = tf.tensor2d([flatFeatures])
        const prediction = this.#model.predict(input)
        try {
            const result = await prediction.data()
            return {
                type: 'prediction',
                grossBuy: result[0],
                grossSell: result[1],
                decision: result[0] > result[1] ? 'BUY' : (result[1] > result[0] ? 'SELL' : 'HOLD'),
                confidence: Math.abs(result[0] - result[1])
            }
        } catch (error) {
            callback(error)
            return
        } finally {
            input.dispose()
            prediction.dispose()
        }
    }

    #getStats (ws) {
        return {
            type: 'stats',
            stats: this.#trainingStats,
            modelParams: this.#model.countParams(),
            config: this.#config
        }
    }
}

// Export for testing
module.exports = TradingTrainingServer

// Start server if run directly
if (require.main === module) {
    const main = async () => {
        process.on('SIGINT', () => {
            console.log('\nShutting down training server...')
            process.exit(0)
        })
        await (TradingTrainingServer.create().catch(console.error))
    }
    main()
}
