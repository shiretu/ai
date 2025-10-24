/**
 * Simple Test Client
 * Connects, sends 3 samples, checks stats, saves model, and exits
 */

const WebSocket = require('ws')

const getWs = async () => {
    const ws = new WebSocket('ws://localhost:8080')
    return new Promise((resolve, reject) => {
        ws.on('open', () => resolve(ws))
        ws.once('error', reject)
    })
}

const doRequest = (ws, request) => {
    return new Promise((resolve, reject) => {
        ws.once('message', (data) => {
            const message = JSON.parse(data.toString())
            resolve(message)
        })
        ws.once('error', reject)
        ws.send(JSON.stringify(request))
    })
}

let sampleNum = 0
const createTrainRequest = (samplesCount) => {
    const result = {
        type: 'train',
        samples: []
    }
    for (let i = 0; i < samplesCount; i++) {
        sampleNum++
        // Create hardcoded sample data with slight variations
        const basePrice = 50000 + (sampleNum * 100)
        const variation = sampleNum * 0.1

        result.samples.push({
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
        })
    }
    return result
}

const work = async () => {
    const ws = await getWs()
    console.log(await doRequest(ws, createTrainRequest(1)))
    console.log(await doRequest(ws, createTrainRequest(10)))
    console.log(await doRequest(ws, createTrainRequest(100)))
    console.log(await doRequest(ws, createTrainRequest(1000)))
    for (let i = 0; i < 4; i++) {
        console.log(await doRequest(ws, createTrainRequest(5000)))
    }
    console.log(JSON.stringify(await doRequest(ws, { type: 'stats' })))
    console.log(await doRequest(ws, { type: 'save' }))
}

work()
