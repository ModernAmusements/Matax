#!/usr/bin/env node

/**
 * Frontend Test for Test Tabs
 * Tests: Health, Detection, Extraction, Reference, Multi, Pose, Eyewear, Viz, Clear
 */

const API_BASE = 'http://localhost:3000/api';

async function testTestTab(tabName) {
    console.log(`\n=== Testing: ${tabName} ===`);
    
    try {
        const response = await fetch(`${API_BASE}/visualizations/${tabName}`);
        const data = await response.json();
        
        console.log('Success:', data.success);
        console.log('Has visualization:', !!data.visualization);
        console.log('Has data:', data.data ? Object.keys(data.data).length : 0, 'keys');
        
        if (data.success && data.data && Object.keys(data.data).length > 0) {
            console.log('Data:', JSON.stringify(data.data, null, 2));
            console.log('✓ PASS');
            return true;
        } else if (data.success && data.visualization) {
            console.log('✓ PASS (has image)');
            return true;
        } else {
            console.log('✗ FAIL - no data or visualization');
            return false;
        }
    } catch (err) {
        console.log('✗ FAIL - Error:', err.message);
        return false;
    }
}

async function runTests() {
    console.log('========================================');
    console.log('TEST TAB VALIDATION');
    console.log('========================================');
    
    const testTabs = [
        'test-health',
        'test-detection', 
        'test-extraction',
        'test-reference',
        'test-multi',
        'test-pose',
        'test-eyewear',
        'test-viz',
        'test-clear',
        'tests'
    ];
    
    let passed = 0;
    let failed = 0;
    
    for (const tab of testTabs) {
        if (await testTestTab(tab)) {
            passed++;
        } else {
            failed++;
        }
    }
    
    console.log('\n========================================');
    console.log(`RESULTS: ${passed}/${testTabs.length} passed`);
    console.log('========================================');
    
    if (failed === 0) {
        console.log('✓ ALL TEST TABS WORKING');
        process.exit(0);
    } else {
        console.log('❌ SOME TEST TABS FAILED');
        process.exit(1);
    }
}

runTests().catch(err => {
    console.error('Test error:', err);
    process.exit(1);
});
