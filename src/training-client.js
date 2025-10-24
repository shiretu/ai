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
                console.log(`   BUY: ${message.buy.toFixed(4)}`)
                console.log(`   SELL: ${message.sell.toFixed(4)}`)
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

        // Example training dataset
        const exampleDataset = [
            {
                features: new Array(1559).fill(0), // Replace with real features
                outcomes: {
                    buy: 1.0, // BUY hit target
                    sell: -1.0 // SELL hit stop
                }
            },
            {
                features: new Array(1559).fill(0.1), // Replace with real features
                outcomes: {
                    buy: -1.0, // BUY hit stop
                    sell: 1.0 // SELL hit target
                }
            },
            {
                features: new Array(1559).fill(0.2), // Replace with real features
                outcomes: {
                    buy: 0.0, // Neither hit target
                    sell: 0.0
                }
            }
        ]

        // Send training data
        await client.sendTrainingData(exampleDataset)

        // Wait for training to complete
        await new Promise(resolve => setTimeout(resolve, 3000))

        // Make a prediction
        const testFeatures = new Array(1559).fill(0.5)
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
