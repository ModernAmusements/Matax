#!/usr/bin/env node

/**
 * Frontend Integration Test for Eyewear Detection
 * Tests the eyewear detection via the API
 */

const API_BASE = 'http://localhost:3000/api';

async function testHealthCheck() {
    console.log('\n=== Test 1: Health Check ===');
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    console.log('Health:', data);
    return data.status === 'ok';
}

async function testEyewearDetection() {
    console.log('\n=== Test 2: Eyewear Detection (No Glasses) ===');
    
    // Load test image
    const imagePath = './test_images/test_subject.jpg';
    const fs = await import('fs');
    const imageBuffer = fs.readFileSync(imagePath);
    const base64 = imageBuffer.toString('base64');
    
    // Step 1: Upload image
    console.log('Uploading image...');
    const detectResponse = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: `data:image/jpeg;base64,${base64}` })
    });
    const detectData = await detectResponse.json();
    console.log('Detect result:', detectData.success ? 'Faces found: ' + detectData.count : 'Failed');
    
    // Step 2: Check eyewear
    console.log('Checking eyewear...');
    const eyewearResponse = await fetch(`${API_BASE}/eyewear`);
    const eyewearData = await eyewearResponse.json();
    console.log('Eyewear result:', eyewearData);
    
    // Verify: should NOT have eyewear for normal face
    if (eyewearData.eyewear && eyewearData.eyewear.has_eyewear) {
        console.log('❌ FAIL: False positive - detected eyewear on face without glasses');
        return false;
    } else {
        console.log('✓ PASS: No false positive');
        return true;
    }
}

async function testVisualization() {
    console.log('\n=== Test 3: Eyewear Visualization ===');
    const response = await fetch(`${API_BASE}/visualizations/eyewear`);
    const data = await response.json();
    console.log('Viz success:', data.success);
    console.log('Viz has image:', !!data.visualization);
    return data.success && !!data.visualization;
}

async function runTests() {
    console.log('========================================');
    console.log('FRONTEND EYWEAR DETECTION TESTS');
    console.log('========================================');
    
    let failed = 0;
    let passed = 0;
    
    // Test 1: Health
    if (await testHealthCheck()) passed++; else failed++;
    
    // Test 2: Eyewear detection (no glasses)
    if (await testEyewearDetection()) passed++; else failed++;
    
    // Test 3: Visualization
    if (await testVisualization()) passed++; else failed++;
    
    console.log('\n========================================');
    console.log(`RESULTS: ${passed} passed, ${failed} failed`);
    console.log('========================================');
    
    if (failed === 0) {
        console.log('✓ ALL TESTS PASSED');
        process.exit(0);
    } else {
        console.log('❌ SOME TESTS FAILED');
        process.exit(1);
    }
}

runTests().catch(err => {
    console.error('Test error:', err);
    process.exit(1);
});
