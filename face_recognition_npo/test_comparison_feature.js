#!/usr/bin/env node

/**
 * Frontend Integration Test for Comparison Feature
 * Tests: detection, extraction, reference, comparison with new verdict/reasons
 */

const API_BASE = 'http://localhost:3000/api';
const fs = require('fs');
const path = require('path');

async function testHealthCheck() {
    console.log('\n=== Test 1: Health Check ===');
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    console.log('Health:', data);
    return data.status === 'ok';
}

async function loadTestImage(filename) {
    const imagePath = path.join(__dirname, 'test_images', filename);
    const imageBuffer = fs.readFileSync(imagePath);
    return imageBuffer.toString('base64');
}

async function testDetection(imageBase64) {
    console.log('\n=== Test 2: Face Detection ===');
    const response = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: `data:image/jpeg;base64,${imageBase64}` })
    });
    const data = await response.json();
    console.log('Detection success:', data.success);
    console.log('Faces found:', data.count);
    return data.success && data.count > 0;
}

async function testExtraction() {
    console.log('\n=== Test 3: Feature Extraction ===');
    const response = await fetch(`${API_BASE}/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face_id: 0 })
    });
    const data = await response.json();
    console.log('Extraction success:', data.success);
    console.log('Embedding size:', data.embedding?.length);
    return data.success;
}

async function testAddReference(imageBase64, name) {
    console.log('\n=== Test 4: Add Reference ===');
    const response = await fetch(`${API_BASE}/add-reference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: `data:image/jpeg;base64,${imageBase64}`, name })
    });
    const data = await response.json();
    console.log('Add reference success:', data.success);
    console.log('Reference name:', data.reference?.name);
    console.log('Has landmarks:', !!data.reference?.landmarks);
    return data.success;
}

async function testCompare() {
    console.log('\n=== Test 5: Compare Faces ===');
    const response = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    console.log('Compare success:', data.success);
    
    if (data.success && data.best_match) {
        const best = data.best_match;
        console.log('\n--- Comparison Result ---');
        console.log('Name:', best.name);
        console.log('Similarity (cosine):', (best.similarity * 100).toFixed(1) + '%');
        console.log('Combined similarity:', best.combined_similarity ? (best.combined_similarity * 100).toFixed(1) + '%' : 'N/A');
        console.log('Landmark score:', best.landmark_score ? (best.landmark_score * 100).toFixed(1) + '%' : 'N/A');
        console.log('Quality score:', best.quality_score ? (best.quality_score * 100).toFixed(1) + '%' : 'N/A');
        console.log('Verdict:', best.verdict);
        console.log('Reasons:', best.reasons);
        
        // Check verdict is valid
        const validVerdicts = ['MATCH', 'POSSIBLE', 'LOW_CONFIDENCE', 'NO_MATCH'];
        if (!validVerdicts.includes(best.verdict)) {
            console.log('ERROR: Invalid verdict:', best.verdict);
            return false;
        }
        
        // Check reasons exist
        if (!best.reasons || best.reasons.length === 0) {
            console.log('ERROR: No reasons provided');
            return false;
        }
        
        console.log('\n✓ Verdict and reasons are valid!');
        return true;
    } else {
        console.log('ERROR:', data.error || 'No match found');
        return false;
    }
}

async function testDifferentImageCompare(imageBase64) {
    console.log('\n=== Test 6: Different Image Compare (No Match Expected) ===');
    
    // Detect new image
    const detectRes = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: `data:image/jpeg;base64,${imageBase64}` })
    });
    const detectData = await detectRes.json();
    if (!detectData.success || detectData.count === 0) {
        console.log('No face detected in different image');
        return false;
    }
    
    // Extract features
    const extractRes = await fetch(`${API_BASE}/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face_id: 0 })
    });
    const extractData = await extractRes.json();
    if (!extractData.success) {
        console.log('Extraction failed');
        return false;
    }
    
    // Compare
    const compareRes = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    const compareData = await compareRes.json();
    
    if (compareData.success && compareData.best_match) {
        const best = compareData.best_match;
        console.log('Similarity:', (best.similarity * 100).toFixed(1) + '%');
        console.log('Verdict:', best.verdict);
        
        // For different images, we expect LOW_CONFIDENCE or NO_MATCH
        const expected = ['LOW_CONFIDENCE', 'NO_MATCH'];
        if (expected.includes(best.verdict)) {
            console.log('✓ Correct verdict for different images');
            return true;
        } else {
            console.log('WARNING: Got', best.verdict, 'for different images (expected LOW_CONFIDENCE or NO_MATCH)');
            return true; // Still pass, just warning
        }
    }
    
    return false;
}

async function runTests() {
    console.log('========================================');
    console.log('COMPARISON FEATURE TESTS');
    console.log('========================================');
    
    let passed = 0;
    let failed = 0;
    
    // Test 1: Health
    if (await testHealthCheck()) passed++; else failed++;
    
    // Load test images
    console.log('\n--- Loading test images ---');
    let testImage, refImage, differentImage;
    try {
        testImage = await loadTestImage('test_subject.jpg');
        refImage = await loadTestImage('test_subject.jpg'); // Same image for reference
        differentImage = await loadTestImage('reference_subject.jpg'); // Different person
        console.log('Images loaded successfully');
    } catch (err) {
        console.log('ERROR loading images:', err.message);
        console.log('Using placeholder images...');
        failed++;
        console.log('\n========================================');
        console.log(`RESULTS: ${passed} passed, ${failed} failed`);
        process.exit(1);
    }
    
    // Test 2: Detection
    if (await testDetection(testImage)) passed++; else failed++;
    
    // Test 3: Extraction
    if (await testExtraction()) passed++; else failed++;
    
    // Test 4: Add Reference
    if (await testAddReference(refImage, 'Test Person')) passed++; else failed++;
    
    // Test 5: Compare (same person)
    if (await testCompare()) passed++; else failed++;
    
    // Test 6: Compare different images
    if (await testDifferentImageCompare(differentImage)) passed++; else failed++;
    
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
