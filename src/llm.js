const tf = require('@tensorflow/tfjs')

class SimpleLLM {
    constructor () {
        this.model = null
        this.inputSize = 5
        this.outputSize = 9 // digits 1-9
        this.hiddenSize = 32
    }

    // Create training data with various patterns
    generateTrainingData (numSamples = 10000) {
        const inputs = []
        const outputs = []

        for (let i = 0; i < numSamples; i++) {
            const pattern = Math.floor(Math.random() * 4) // 4 different pattern types

            let sequence

            switch (pattern) {
                case 0: // Arithmetic progression
                    sequence = this.generateArithmetic()
                    break
                case 1: // Repeating pattern
                    sequence = this.generateRepeating()
                    break
                case 2: // Fibonacci-like
                    sequence = this.generateFibonacci()
                    break
                case 3: // Random with slight bias
                    sequence = this.generateBiased()
                    break
            }

            // Normalize inputs to 0-1 range (1-9 becomes 0-8, then divide by 8)
            const inputSequence = []
            for (let j = 0; j < 5; j++) {
                inputSequence.push((sequence[j] - 1) / 8)
            }
            const nextNum = sequence[5]

            inputs.push(inputSequence)

            // One-hot encode output (1-9 becomes index 0-8)
            const oneHot = new Array(9).fill(0)
            oneHot[nextNum - 1] = 1
            outputs.push(oneHot)
        }

        return {
            inputs: tf.tensor2d(inputs),
            outputs: tf.tensor2d(outputs)
        }
    }

    generateArithmetic () {
        const start = Math.floor(Math.random() * 5) + 1
        const diff = Math.floor(Math.random() * 3) + 1
        const sequence = []

        for (let i = 0; i < 6; i++) {
            let val = start + diff * i
            if (val > 9) val = ((val - 1) % 9) + 1
            sequence.push(val)
        }
        return sequence
    }

    generateRepeating () {
        const patternLength = Math.floor(Math.random() * 3) + 2 // 2-4 length
        const pattern = []
        for (let i = 0; i < patternLength; i++) {
            pattern.push(Math.floor(Math.random() * 9) + 1)
        }

        const sequence = []
        for (let i = 0; i < 6; i++) {
            sequence.push(pattern[i % patternLength])
        }
        return sequence
    }

    generateFibonacci () {
        let a = Math.floor(Math.random() * 4) + 1
        let b = Math.floor(Math.random() * 4) + 1
        const sequence = [a, b]

        for (let i = 2; i < 6; i++) {
            let next = (a + b)
            if (next > 9) next = ((next - 1) % 9) + 1
            sequence.push(next)
            a = b
            b = next
        }
        return sequence
    }

    generateBiased () {
        const sequence = []
        let prev = Math.floor(Math.random() * 9) + 1
        sequence.push(prev)

        for (let i = 1; i < 6; i++) {
            // 50% chance to be close to previous number
            if (Math.random() < 0.5) {
                let next = prev + Math.floor(Math.random() * 3) - 1
                if (next < 1) next = 1
                if (next > 9) next = 9
                sequence.push(next)
                prev = next
            } else {
                prev = Math.floor(Math.random() * 9) + 1
                sequence.push(prev)
            }
        }
        return sequence
    }

    // Build the neural network model
    buildModel () {
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [this.inputSize],
                    units: this.hiddenSize,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.hiddenSize,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.outputSize,
                    activation: 'softmax'
                })
            ]
        })

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        })

        console.log('Model built!')
        this.model.summary()
    }

    // Train the model
    async trainModel (epochs = 100) {
        console.log('Generating training data...')
        const { inputs, outputs } = this.generateTrainingData(10000)

        console.log('Starting training...')
        const history = await this.model.fit(inputs, outputs, {
            epochs,
            batchSize: 32,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0) {
                        console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`)
                    }
                }
            }
        })

        inputs.dispose()
        outputs.dispose()

        console.log('Training completed!')
        return history
    }

    // Predict next number in sequence
    predict (sequence) {
        if (sequence.length !== 5) {
            throw new Error('Input sequence must be exactly 5 numbers')
        }

        // Normalize input
        const normalizedInput = sequence.map(x => (x - 1) / 8)
        const inputTensor = tf.tensor2d([normalizedInput])

        // Get prediction
        const prediction = this.model.predict(inputTensor)
        const probabilities = prediction.dataSync()

        // Find the most likely digit (add 1 to convert from 0-8 to 1-9)
        const predictedIndex = probabilities.indexOf(Math.max(...probabilities))
        const predictedNumber = predictedIndex + 1

        inputTensor.dispose()
        prediction.dispose()

        return {
            predicted: predictedNumber,
            confidence: probabilities[predictedIndex],
            allProbabilities: Array.from(probabilities).map((prob, idx) => ({
                number: idx + 1,
                probability: prob
            }))
        }
    }

    // Export training data for Python
    exportTrainingData (numSamples = 10000, filename = 'training_data.json') {
        console.log(`Generating ${numSamples} training samples for export...`)

        const exportData = {
            metadata: {
                inputSize: this.inputSize,
                outputSize: this.outputSize,
                numSamples,
                generated: new Date().toISOString()
            },
            samples: []
        }

        for (let i = 0; i < numSamples; i++) {
            const pattern = Math.floor(Math.random() * 4)
            let sequence

            switch (pattern) {
                case 0:
                    sequence = this.generateArithmetic()
                    break
                case 1:
                    sequence = this.generateRepeating()
                    break
                case 2:
                    sequence = this.generateFibonacci()
                    break
                case 3:
                    sequence = this.generateBiased()
                    break
            }

            const input = sequence.slice(0, 5)
            const target = sequence[5]

            exportData.samples.push({
                input,
                target,
                pattern
            })
        }

        const fs = require('fs')
        fs.writeFileSync(filename, JSON.stringify(exportData, null, 2))
        console.log(`Training data exported to ${filename}`)
        return exportData
    }

    // Export model architecture for Python
    exportModelArchitecture (filename = 'model_architecture.json') {
        const architecture = {
            metadata: {
                framework: 'tensorflow',
                inputShape: [this.inputSize],
                outputShape: [this.outputSize],
                created: new Date().toISOString()
            },
            layers: [
                {
                    type: 'Dense',
                    units: this.hiddenSize,
                    activation: 'relu',
                    inputShape: [this.inputSize]
                },
                {
                    type: 'Dense',
                    units: this.hiddenSize,
                    activation: 'relu'
                },
                {
                    type: 'Dense',
                    units: this.outputSize,
                    activation: 'softmax'
                }
            ],
            compile: {
                optimizer: 'adam',
                loss: 'sparse_categorical_crossentropy',
                metrics: ['accuracy'],
                learningRate: 0.001
            }
        }

        const fs = require('fs')
        fs.writeFileSync(filename, JSON.stringify(architecture, null, 2))
        console.log(`Model architecture exported to ${filename}`)
        return architecture
    }

    // Test the model with some examples
    runTests () {
        console.log('\n=== Testing the trained model ===')

        const testCases = [
            [1, 2, 3, 4, 5], // Arithmetic +1
            [2, 4, 6, 8, 1], // Arithmetic +2 (wrapping)
            [1, 1, 2, 3, 5], // Fibonacci-like
            [1, 2, 1, 2, 1], // Repeating pattern
            [5, 5, 5, 5, 5], // Constant
            [9, 8, 7, 6, 5], // Arithmetic -1
            [1, 3, 5, 7, 9] // Arithmetic +2
        ]

        testCases.forEach((testCase, idx) => {
            const result = this.predict(testCase)
            console.log(`Test ${idx + 1}: [${testCase.join(', ')}] -> ${result.predicted} (confidence: ${(result.confidence * 100).toFixed(1)}%)`)
        })
    }
}

// Main execution
async function main () {
    console.log('Creating Simple LLM for number sequence prediction...')

    const llm = new SimpleLLM()
    llm.buildModel()

    // Export data and architecture for Python
    console.log('\n=== Exporting for Python ===')
    llm.exportTrainingData(10000, 'training_data.json')
    llm.exportModelArchitecture('model_architecture.json')

    await llm.trainModel(50) // Train for 50 epochs

    llm.runTests()

    console.log('\nYou can now use llm.predict([1, 2, 3, 4, 5]) to predict the next number!')
    console.log('Data exported for Python training - check training_data.json and model_architecture.json')
}// Export for use in other files
module.exports = { SimpleLLM }

// Run if this file is executed directly
if (require.main === module) {
    main().catch(console.error)
}
