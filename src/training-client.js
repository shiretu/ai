/**
 * Training Client Example
 * Demonstrates how to send training data to the training server
 */

const WebSocket = require('ws')

class TrainingClient {
    constructor (serverUrl = 'ws://localhost:8080') {
        this.serverUrl = serverUrl
        this.ws = null
        this.connected = false
    }

    async connect () {
        return new Promise((resolve, reject) => {
            console.log(`🔗 Connecting to training server: ${this.serverUrl}`)

            this.ws = new WebSocket(this.serverUrl)

            this.ws.on('open', () => {
                console.log('✅ Connected to training server')
                this.connected = true
                resolve()
            })

            this.ws.on('message', (data) => {
                this.handleMessage(JSON.parse(data.toString()))
            })

            this.ws.on('error', (error) => {
                console.error('❌ WebSocket error:', error)
                reject(error)
            })

            this.ws.on('close', () => {
                console.log('🔌 Connection closed')
                this.connected = false
            })
        })
    }

    handleMessage (message) {
        switch (message.type) {
            case 'connected':
                console.log('🎯 Server ready:', message.message)
                console.log(`📊 Model parameters: ${message.modelParams}`)
                break

            case 'training_started':
                console.log(`🚀 Training started with ${message.samples} samples`)
                break

            case 'training_progress':
                console.log(`📈 Epoch ${message.epoch}: Loss=${message.loss.toFixed(6)}, MAE=${message.mae.toFixed(6)}`)
                break

            case 'training_completed':
                console.log('✅ Training completed!')
                console.log(`   Samples: ${message.samples}`)
                console.log(`   Loss: ${message.loss.toFixed(6)}`)
                console.log(`   Duration: ${message.duration}ms`)
                console.log(`   Total samples trained: ${message.totalSamples}`)
                break

            case 'prediction':
                console.log('🔮 Prediction:')
                console.log(`   Gross BUY: ${message.grossBuy.toFixed(4)}`)
                console.log(`   Gross SELL: ${message.grossSell.toFixed(4)}`)
                console.log(`   Decision: ${message.decision}`)
                console.log(`   Confidence: ${message.confidence.toFixed(4)}`)
                break

            case 'model_saved':
                console.log(`💾 Model saved to: ${message.path}`)
                break

            case 'error':
                console.error(`❌ Server error: ${message.message}`)
                break

            default:
                console.log('📨 Received:', message)
        }
    }

    async sendTrainingData (dataset) {
        if (!this.connected) {
            throw new Error('Not connected to server')
        }

        console.log(`📤 Sending ${dataset.length} training samples...`)

        this.ws.send(JSON.stringify({
            type: 'train',
            dataset
        }))
    }

    async makePrediction (features) {
        if (!this.connected) {
            throw new Error('Not connected to server')
        }

        this.ws.send(JSON.stringify({
            type: 'predict',
            features
        }))
    }

    async saveModel () {
        if (!this.connected) {
            throw new Error('Not connected to server')
        }

        this.ws.send(JSON.stringify({
            type: 'save'
        }))
    }

    async getStats () {
        if (!this.connected) {
            throw new Error('Not connected to server')
        }

        this.ws.send(JSON.stringify({
            type: 'stats'
        }))
    }

    disconnect () {
        if (this.ws) {
            this.ws.close()
        }
    }
}

// Example usage
async function example () {
    const client = new TrainingClient()

    try {
        // Connect to server
        await client.connect()

        // Wait a moment for connection to stabilize
        await new Promise(resolve => setTimeout(resolve, 1000))

        // Example training dataset with nested feature structure
        const exampleDataset = [
            {
                features: {
                    candles: {
                        opens: new Array(120).fill(50000),
                        highs: new Array(120).fill(51000),
                        lows: new Array(120).fill(49000),
                        closes: new Array(120).fill(50500),
                        volumes: new Array(120).fill(100),
                        timestamps: new Array(120).fill(Date.now()),
                        colors: new Array(120).fill(1),
                        bodySizes: new Array(120).fill(500)
                    },
                    studies: {
                        sma9: new Array(120).fill(50200),
                        sma12: new Array(120).fill(50100),
                        sma21: new Array(120).fill(50000),
                        ema9: new Array(120).fill(50250),
                        ema12: new Array(120).fill(50150),
                        ema21: new Array(120).fill(50050),
                        rsi14: new Array(120).fill(55)
                    },
                    patterns: {
                        single: new Array(119).fill(0),
                        sliding: new Array(118).fill(0)
                    },
                    global: {
                        candleDuration: 1,
                        windowSize: 120,
                        grossProfitTarget: 0.7,
                        grossStopLoss: 0.4,
                        positionSize: 100,
                        fees: 0.2
                    }
                },
                outcomes: {
                    grossBuy: 0.73, // Gross profit with overshoot
                    grossSell: -0.47 // Gross loss with overshoot
                }
            },
            {
                features: {
                    candles: {
                        opens: new Array(120).fill(51000),
                        highs: new Array(120).fill(52000),
                        lows: new Array(120).fill(50000),
                        closes: new Array(120).fill(50200),
                        volumes: new Array(120).fill(120),
                        timestamps: new Array(120).fill(Date.now()),
                        colors: new Array(120).fill(-1),
                        bodySizes: new Array(120).fill(800)
                    },
                    studies: {
                        sma9: new Array(120).fill(50800),
                        sma12: new Array(120).fill(50700),
                        sma21: new Array(120).fill(50600),
                        ema9: new Array(120).fill(50850),
                        ema12: new Array(120).fill(50750),
                        ema21: new Array(120).fill(50650),
                        rsi14: new Array(120).fill(45)
                    },
                    patterns: {
                        single: new Array(119).fill(0),
                        sliding: new Array(118).fill(0)
                    },
                    global: {
                        candleDuration: 1,
                        windowSize: 120,
                        grossProfitTarget: 0.7,
                        grossStopLoss: 0.4,
                        positionSize: 100,
                        fees: 0.2
                    }
                },
                outcomes: {
                    grossBuy: -0.18, // Gross loss with undershoot
                    grossSell: 0.51 // Gross profit with overshoot
                }
            },
            {
                features: {
                    candles: {
                        opens: new Array(120).fill(50500),
                        highs: new Array(120).fill(50800),
                        lows: new Array(120).fill(50200),
                        closes: new Array(120).fill(50550),
                        volumes: new Array(120).fill(80),
                        timestamps: new Array(120).fill(Date.now()),
                        colors: new Array(120).fill(1),
                        bodySizes: new Array(120).fill(300)
                    },
                    studies: {
                        sma9: new Array(120).fill(50400),
                        sma12: new Array(120).fill(50350),
                        sma21: new Array(120).fill(50300),
                        ema9: new Array(120).fill(50450),
                        ema12: new Array(120).fill(50400),
                        ema21: new Array(120).fill(50350),
                        rsi14: new Array(120).fill(50)
                    },
                    patterns: {
                        single: new Array(119).fill(0),
                        sliding: new Array(118).fill(0)
                    },
                    global: {
                        candleDuration: 1,
                        windowSize: 120,
                        grossProfitTarget: 0.7,
                        grossStopLoss: 0.4,
                        positionSize: 100,
                        fees: 0.2
                    }
                },
                outcomes: {
                    grossBuy: 0.0, // Neither hit target
                    grossSell: 0.0
                }
            }
        ]

        // Send training data
        await client.sendTrainingData(exampleDataset)

        // Wait for training to complete
        await new Promise(resolve => setTimeout(resolve, 3000))

        // Make a prediction with nested features
        const testFeatures = {
            candles: {
                opens: new Array(120).fill(50300),
                highs: new Array(120).fill(50600),
                lows: new Array(120).fill(50000),
                closes: new Array(120).fill(50400),
                volumes: new Array(120).fill(90),
                timestamps: new Array(120).fill(Date.now()),
                colors: new Array(120).fill(1),
                bodySizes: new Array(120).fill(400)
            },
            studies: {
                sma9: new Array(120).fill(50300),
                sma12: new Array(120).fill(50250),
                sma21: new Array(120).fill(50200),
                ema9: new Array(120).fill(50350),
                ema12: new Array(120).fill(50300),
                ema21: new Array(120).fill(50250),
                rsi14: new Array(120).fill(52)
            },
            patterns: {
                single: new Array(119).fill(0),
                sliding: new Array(118).fill(0)
            },
            global: {
                candleDuration: 1,
                windowSize: 120,
                grossProfitTarget: 0.7,
                grossStopLoss: 0.4,
                positionSize: 100,
                fees: 0.2
            }
        }
        await client.makePrediction(testFeatures)

        // Save the model
        await client.saveModel()

        // Get training stats
        await client.getStats()
    } catch (error) {
        console.error('❌ Example error:', error)
    } finally {
        // Clean disconnect
        setTimeout(() => {
            client.disconnect()
        }, 2000)
    }
}

module.exports = TrainingClient

// Run example if called directly
if (require.main === module) {
    example().catch(console.error)
}
