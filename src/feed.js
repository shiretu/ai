/**
 * Simple Test Client
 * Connects, sends 3 samples, checks stats, saves model, and exits
 */

const WebSocket = require('ws')

class TestClient {
    constructor () {
        this.ws = null
        this.sampleCount = 0
        this.targetSamples = 3
    }

    async run () {
        console.log('Starting simple test client...')

        // Connect to server
        this.ws = new WebSocket('ws://localhost:8080')

        this.ws.on('open', () => {
            console.log('Connected to training server')
            this.start()
        })

        this.ws.on('message', (data) => {
            const message = JSON.parse(data.toString())
            this.handleMessage(message)
        })

        this.ws.on('error', (error) => {
            console.error('WebSocket error:', error)
            process.exit(1)
        })

        this.ws.on('close', () => {
            console.log('Connection closed')
            process.exit(0)
        })
    }

    handleMessage (message) {
        switch (message.type) {
            case 'connected':
                console.log(`Server ready with ${message.modelParams} parameters`)
                break

            case 'training_started':
                console.log(`Training sample ${this.sampleCount + 1}/${this.targetSamples}...`)
                break

            case 'training_progress':
                console.log(`   Epoch ${message.epoch}: Loss=${message.loss.toFixed(6)}`)
                break

            case 'training_completed':
                this.sampleCount++
                console.log(`   Sample ${this.sampleCount} completed - Loss: ${message.loss.toFixed(6)}`)

                if (this.sampleCount < this.targetSamples) {
                    // Send next sample
                    setTimeout(() => this.sendNextSample(), 100)
                } else {
                    // All samples sent, get final stats and save
                    setTimeout(() => this.finalize(), 500)
                }
                break

            case 'error':
                console.error(`Server error: ${message.message}`)
                this.ws.close()
                break

            default:
                if (message.totalSamples !== undefined) {
                    console.log(`Stats: ${message.totalSamples} total samples, avg loss: ${message.averageLoss.toFixed(6)}`)
                } else if (message.path) {
                    console.log(`Model saved to: ${message.path}`)
                    this.ws.close()
                } else {
                    console.log('Response:', message.type || JSON.stringify(message))
                }
        }
    }

    async start () {
        console.log('\nGetting initial stats...')
        this.ws.send(JSON.stringify({ type: 'stats' }))

        // Wait a bit then start sending samples
        setTimeout(() => {
            console.log('\nStarting to send 3 training samples...')
            this.sendNextSample()
        }, 500)
    }

    sendNextSample () {
        const sampleData = this.createSample(this.sampleCount + 1)

        this.ws.send(JSON.stringify({
            type: 'train',
            sample: sampleData
        }))
    }

    createSample (sampleNum) {
        // Create hardcoded sample data with slight variations
        const basePrice = 50000 + (sampleNum * 100)
        const variation = sampleNum * 0.1

        return {
            features: {
                candles: {
                    opens: new Array(120).fill(basePrice),
                    highs: new Array(120).fill(basePrice + 500),
                    lows: new Array(120).fill(basePrice - 500),
                    closes: new Array(120).fill(basePrice + (sampleNum * 50)),
                    volumes: new Array(120).fill(100 + sampleNum * 10),
                    timestamps: new Array(120).fill(Date.now()),
                    colors: new Array(120).fill(sampleNum % 2 === 0 ? 1 : -1),
                    bodySizes: new Array(120).fill(200 + sampleNum * 100)
                },
                studies: {
                    sma9: new Array(120).fill(basePrice + 100),
                    sma12: new Array(120).fill(basePrice + 50),
                    sma21: new Array(120).fill(basePrice),
                    ema9: new Array(120).fill(basePrice + 150),
                    ema12: new Array(120).fill(basePrice + 100),
                    ema21: new Array(120).fill(basePrice + 50),
                    rsi14: new Array(120).fill(50 + sampleNum * 5)
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
                grossBuy: variation + (sampleNum === 1 ? 0.8 : sampleNum === 2 ? -0.3 : 0.0),
                grossSell: variation + (sampleNum === 1 ? -0.2 : sampleNum === 2 ? 0.6 : 0.0)
            }
        }
    }

    finalize () {
        console.log('\nGetting final stats...')
        this.ws.send(JSON.stringify({ type: 'stats' }))

        setTimeout(() => {
            console.log('\nSaving model...')
            this.ws.send(JSON.stringify({ type: 'save' }))
        }, 500)
    }
}

// Run the test client
const client = new TestClient()
client.run().catch(console.error)
