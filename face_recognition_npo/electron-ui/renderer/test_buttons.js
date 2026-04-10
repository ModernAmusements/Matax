// ============================================================================
// COMPREHENSIVE FRONTEND BUTTON TEST
// Tests all onclick handlers for errors
// ============================================================================

(function() {
    'use strict';

    var results = {
        passed: 0,
        failed: 0,
        errors: []
    };

    // Collect all onclick handlers from HTML
    var handlers = [
        // Sidebar
        { name: 'jumpToStep(6)', fn: function() { return jumpToStep(6); }},
        { name: 'jumpToStep(1)', fn: function() { return jumpToStep(1); }},
        { name: 'jumpToStep(2)', fn: function() { return jumpToStep(2); }},
        { name: 'jumpToStep(3)', fn: function() { return jumpToStep(3); }},
        { name: 'jumpToStep(4)', fn: function() { return jumpToStep(4); }},
        
        // Window controls
        { name: 'closeWindow', fn: function() { return closeWindow(); }},
        { name: 'minimizeWindow', fn: function() { return minimizeWindow(); }},
        { name: 'maximizeWindow', fn: function() { return maximizeWindow(); }},
        { name: 'toggleSidebar', fn: function() { return toggleSidebar(); }},
        
        // Step navigation
        { name: 'scrollToStep1', fn: function() { return scrollToStep1(); }},
        { name: 'scrollToSection(webcamStep)', fn: function() { return scrollToSection('webcamStep'); }},
        { name: 'scrollToSection(step4)', fn: function() { return scrollToSection('step4'); }},
        { name: 'scrollToSection(step6)', fn: function() { return scrollToSection('step6'); }},
        
        // Image selection
        { name: 'selectImage', fn: function() { return selectImage(); }},
        { name: 'selectImageForStep5', fn: function() { return selectImageForStep5(); }},
        
        // Cache
        { name: 'clearAllCache', fn: function() { return clearAllCache(); }},
        
        // Library
        { name: 'startWebcamForLibrary', fn: function() { return startWebcamForLibrary(); }},
        { name: 'searchLibraryByName', fn: function() { return searchLibraryByName(''); }},
        { name: 'renderLibraryGrid', fn: function() { return renderLibraryGrid(); }},
        { name: 'matchWithLibraryImage(null)', fn: function() { return matchWithLibraryImage(null); }},
        
        // Compare
        { name: 'compareWithUploadedImage', fn: function() { return compareWithUploadedImage(); }},
        
        // Webcam
        { name: 'startWebcam', fn: function() { return startWebcam(); }},
        { name: 'captureWebcam', fn: function() { return captureWebcam(); }},
        { name: 'startAutoCapture', fn: function() { return startAutoCapture(); }},
        { name: 'stopWebcam', fn: function() { return stopWebcam(); }},
        { name: 'toggleMeshOverlay', fn: function() { return toggleMeshOverlay(); }},
        { name: 'startProfileCapture', fn: function() { return startProfileCapture(); }},
        { name: 'useForMatching', fn: function() { return useForMatching(); }},
        
        // Reference
        { name: 'addReference', fn: function() { return addReference(); }},
        { name: 'showLibraryModal', fn: function() { return showLibraryModal(); }},
        { name: 'showUploadedAsReference', fn: function() { return showUploadedAsReference(); }},
        { name: 'addUploadedToLibrary', fn: function() { return addUploadedToLibrary(); }},
        
        // Compare buttons
        { name: 'compareFaces', fn: function() { return compareFaces(); }},
        { name: 'compareWithLibrary', fn: function() { return compareWithLibrary(); }},
        
        // Reference details
        { name: 'hideReferenceDetails', fn: function() { return hideReferenceDetails(); }},
        
        // Library modal
        { name: 'closeLibraryModal', fn: function() { return closeLibraryModal(); }},
        { name: 'saveToLibrary', fn: function() { return saveToLibrary(); }},
        { name: 'closeLibraryInfoPopup', fn: function() { return closeLibraryInfoPopup(); }},
        
        // Viz tabs (sample)
        { name: 'showVisualization(detection)', fn: function() { return showVisualization('detection'); }},
        { name: 'showVisualization(landmarks)', fn: function() { return showVisualization('landmarks'); }},
        
        // Terminal
        { name: 'toggleTerminal', fn: function() { return toggleTerminal(); }},
        
        // Step buttons
        { name: 'detectFaces', fn: function() { return detectFaces(); }},
        { name: 'extractFeatures', fn: function() { return extractFeatures(); }},
    ];

    // Error handler
    window.onerror = function(msg, url, line, col, error) {
        results.failed++;
        results.errors.push({
            handler: 'GLOBAL',
            error: msg,
            line: line
        });
        console.error('[TEST ERROR]', msg);
        return false;
    };

    // Catch console errors
    var originalError = console.error;
    console.error = function() {
        results.failed++;
        var args = Array.prototype.slice.call(arguments);
        results.errors.push({
            handler: 'CONSOLE',
            error: args.join(' ')
        });
        originalError.apply(console, arguments);
    };

    // Run tests
    function runTests() {
        console.log('========================================');
        console.log('COMPREHENSIVE BUTTON TEST');
        console.log('========================================');
        
        handlers.forEach(function(test) {
            try {
                test.fn();
                results.passed++;
                console.log('[PASS]', test.name);
            } catch (e) {
                results.failed++;
                results.errors.push({
                    handler: test.name,
                    error: e.message
                });
                console.log('[FAIL]', test.name, '-', e.message);
            }
        });

        // Restore console
        console.error = originalError;

        // Report
        console.log('========================================');
        console.log('RESULTS:', results.passed, 'passed,', results.failed, 'failed');
        console.log('========================================');
        
        if (results.errors.length > 0) {
            console.log('ERRORS:');
            results.errors.forEach(function(err) {
                console.log(' -', err.handler + ':', err.error);
            });
        }

        return results;
    }

    // Expose to run manually
    window.runButtonTests = runTests;

    // Auto-run after DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runTests);
    } else {
        setTimeout(runTests, 1000); // Wait for scripts to load
    }

})();
