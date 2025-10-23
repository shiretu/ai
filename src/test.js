// Simple test function
function runTests () {
    console.log('Running tests...')

    // Simple test example
    const result = 2 + 2
    const expected = 4

    if (result === expected) {
        console.log('✓ Basic math test passed')
    } else {
        console.log('✗ Basic math test failed')
        console.log(`Expected: ${expected}`)
        console.log(`Got: ${result}`)
    }

    console.log('Tests completed!')
}

// Run the tests
runTests()
