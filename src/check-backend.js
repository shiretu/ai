const tf = require('@tensorflow/tfjs')

async function checkBackend () {
    // Initialize TensorFlow.js
    await tf.ready()

    console.log('=== TensorFlow.js Backend Information ===')
    console.log('Current backend:', tf.getBackend())

    // Check platform info
    console.log('Platform:', process.platform)
    console.log('Arch:', process.arch)

    // Check if we're using CPU or potentially GPU
    const backend = tf.getBackend()
    if (backend === 'webgl') {
        console.log('✅ WebGL backend - Using GPU acceleration!')
    } else if (backend === 'cpu') {
        console.log('❌ CPU backend - No hardware acceleration')
    } else if (backend === 'tensorflow') {
        console.log('✅ TensorFlow backend - Potentially using hardware acceleration')
    } else {
        console.log('Backend:', backend)
    }

    console.log('\n=== Available Backends ===')
    // Try to list what's available
    try {
        console.log('tf.ENV.flags:', Object.keys(tf.env().flags || {}))
    } catch (e) {
        console.log('Could not access environment flags')
    }

    // Test a simple operation and time it
    console.log('\n=== Performance Test ===')
    const start = Date.now()

    // Create some tensors and do operations
    const a = tf.randomNormal([1000, 1000])
    const b = tf.randomNormal([1000, 1000])
    const c = tf.matMul(a, b)
    await c.data() // Wait for computation to complete

    const elapsed = Date.now() - start
    console.log(`Matrix multiplication (1000x1000) took: ${elapsed}ms`)

    // Clean up
    a.dispose()
    b.dispose()
    c.dispose()

    console.log('\n=== Memory Info ===')
    console.log('Memory info:', tf.memory())
}

checkBackend().catch(console.error)
