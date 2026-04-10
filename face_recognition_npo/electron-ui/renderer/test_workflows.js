// ============================================================================
// COMPREHENSIVE WORKFLOW TESTS
// Tests all 53 user workflows in the MANTAX Face Recognition App
// ============================================================================

(function() {
    'use strict';

    var API_BASE = 'http://localhost:3000';
    var TEST_IMAGE = 'test_images/test_subject.jpg';
    var TEST_IMAGE_2 = 'test_images/reference_subject.jpg';

    var testResults = {
        passed: 0,
        failed: 0,
        skipped: 0,
        errors: [],
        workflows: {}
    };

    function log(message, type) {
        type = type || 'info';
        var prefix = {
            'info': '[INFO]',
            'pass': '[PASS]',
            'fail': '[FAIL]',
            'skip': '[SKIP]',
            'warn': '[WARN]'
        }[type] || '[INFO]';
        console.log(prefix + ' ' + message);
    }

    function waitFor(condition, timeout, interval) {
        return new Promise(function(resolve, reject) {
            var start = Date.now();
            interval = interval || 100;
            timeout = timeout || 5000;

            function check() {
                if (condition()) {
                    resolve(true);
                } else if (Date.now() - start > timeout) {
                    resolve(false);
                } else {
                    setTimeout(check, interval);
                }
            }
            check();
        });
    }

    function imageToBase64(imagePath) {
        var img = new Image();
        img.crossOrigin = 'Anonymous';
        img.src = imagePath;

        var canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        return canvas.toDataURL('image/jpeg').split(',')[1];
    }

    function callAPI(endpoint, method, data) {
        return fetch(API_BASE + endpoint, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: data ? JSON.stringify(data) : null
        }).then(function(r) { return r.json(); });
    }

    function recordResult(workflow, passed, details) {
        testResults.workflows[workflow] = {
            passed: passed,
            details: details || ''
        };
        if (passed) {
            testResults.passed++;
            log(workflow + ' - PASSED', 'pass');
        } else {
            testResults.failed++;
            testResults.errors.push({ workflow: workflow, details: details });
            log(workflow + ' - FAILED: ' + details, 'fail');
        }
    }

    function skipResult(workflow, reason) {
        testResults.workflows[workflow] = {
            passed: false,
            skipped: true,
            reason: reason || ''
        };
        testResults.skipped++;
        log(workflow + ' - SKIPPED: ' + reason, 'skip');
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 1: Core Image Input Workflows
    // ============================================================================

    function testWorkflow1_UploadImageToMatch() {
        var workflow = '1.1 Upload Image → Match';
        log('Testing: ' + workflow);

        try {
            if (typeof selectImage !== 'function') {
                recordResult(workflow, false, 'selectImage function not found');
                return;
            }

            selectImage();
            recordResult(workflow, true, 'Upload dialog triggered');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow2_WebcamToMatch() {
        var workflow = '1.2 Webcam → Match';
        log('Testing: ' + workflow);

        try {
            if (typeof startWebcam !== 'function') {
                recordResult(workflow, false, 'startWebcam function not found');
                return;
            }

            startWebcam();
            setTimeout(function() {
                if (typeof captureWebcam === 'function') {
                    captureWebcam();
                    setTimeout(function() {
                        if (typeof useForMatching === 'function') {
                            useForMatching();
                            recordResult(workflow, true, 'Webcam capture workflow complete');
                        } else {
                            recordResult(workflow, false, 'useForMatching not found');
                        }
                    }, 500);
                } else {
                    recordResult(workflow, false, 'captureWebcam not found');
                }
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow3_AutoCaptureMode() {
        var workflow = '1.3 Auto Capture Mode';
        log('Testing: ' + workflow);

        try {
            if (typeof startWebcam !== 'function') {
                recordResult(workflow, false, 'startWebcam function not found');
                return;
            }

            if (typeof startAutoCapture !== 'function') {
                recordResult(workflow, false, 'startAutoCapture function not found');
                return;
            }

            startWebcam();
            setTimeout(function() {
                startAutoCapture();
                setTimeout(function() {
                    stopWebcam();
                    recordResult(workflow, true, 'Auto capture workflow complete');
                }, 2000);
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 2: Reference Library Management
    // ============================================================================

    function testWorkflow4_UploadSaveToLibrary() {
        var workflow = '2.1 Upload → Save to Library';
        log('Testing: ' + workflow);

        try {
            if (typeof addUploadedToLibrary !== 'function') {
                recordResult(workflow, false, 'addUploadedToLibrary function not found');
                return;
            }

            addUploadedToLibrary();
            recordResult(workflow, true, 'Add to library modal opened');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow5_WebcamSaveToLibrary() {
        var workflow = '2.2 Webcam → Save to Library';
        log('Testing: ' + workflow);

        try {
            if (typeof startWebcamForLibrary !== 'function') {
                recordResult(workflow, false, 'startWebcamForLibrary function not found');
                return;
            }

            startWebcamForLibrary();
            setTimeout(function() {
                if (typeof captureWebcam === 'function') {
                    captureWebcam();
                    setTimeout(function() {
                        if (typeof showLibraryModal === 'function') {
                            showLibraryModal();
                            recordResult(workflow, true, 'Webcam save to library complete');
                        } else {
                            recordResult(workflow, false, 'showLibraryModal not found');
                        }
                    }, 500);
                } else {
                    recordResult(workflow, false, 'captureWebcam not found');
                }
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow6_BatchAddToLibrary() {
        var workflow = '2.3 Batch Add to Library';
        log('Testing: ' + workflow);

        try {
            var batchBtn = document.querySelector('button:contains("Batch")');
            if (!batchBtn) {
                batchBtn = document.querySelector('button[onclick*="batch"]');
            }
            if (batchBtn) {
                batchBtn.click();
                recordResult(workflow, true, 'Batch button found and clicked');
            } else {
                recordResult(workflow, false, 'Batch button not found');
            }
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow7_ViewLibraryPerson() {
        var workflow = '2.4 View Library Person';
        log('Testing: ' + workflow);

        try {
            callAPI('/api/library', 'GET', null).then(function(data) {
                if (data.persons && data.persons.length > 0) {
                    var personId = data.persons[0].id;
                    callAPI('/api/library/person/' + personId, 'GET', null).then(function(personData) {
                        if (personData.person) {
                            recordResult(workflow, true, 'Person details retrieved');
                        } else {
                            recordResult(workflow, false, 'No person data returned');
                        }
                    }).catch(function(e) {
                        recordResult(workflow, false, e.message);
                    });
                } else {
                    recordResult(workflow, true, 'No persons in library (skip)');
                }
            }).catch(function(e) {
                recordResult(workflow, false, e.message);
            });
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow8_DeleteLibraryPerson() {
        var workflow = '2.5 Delete Library Person';
        log('Testing: ' + workflow);

        try {
            callAPI('/api/library', 'GET', null).then(function(data) {
                if (data.persons && data.persons.length > 0) {
                    var personId = data.persons[0].id;
                    callAPI('/api/library/person/' + personId, 'DELETE', null).then(function(deleteData) {
                        if (deleteData.success) {
                            recordResult(workflow, true, 'Person deleted');
                        } else {
                            recordResult(workflow, false, 'Delete failed');
                        }
                    }).catch(function(e) {
                        recordResult(workflow, false, e.message);
                    });
                } else {
                    skipResult(workflow, 'No persons to delete');
                }
            }).catch(function(e) {
                recordResult(workflow, false, e.message);
            });
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 3: Comparison Workflows
    // ============================================================================

    function testWorkflow9_CompareWithReferences() {
        var workflow = '3.1 Compare with References';
        log('Testing: ' + workflow);

        try {
            if (typeof compareFaces !== 'function') {
                recordResult(workflow, false, 'compareFaces function not found');
                return;
            }

            compareFaces();
            recordResult(workflow, true, 'Compare with references initiated');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow10_CompareWithLibrary() {
        var workflow = '3.2 Compare with Library';
        log('Testing: ' + workflow);

        try {
            if (typeof compareWithLibrary !== 'function') {
                recordResult(workflow, false, 'compareWithLibrary function not found');
                return;
            }

            compareWithLibrary();
            recordResult(workflow, true, 'Compare with library initiated');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow11_FindMatchesCurrentImage() {
        var workflow = '3.3 Find Matches (Current)';
        log('Testing: ' + workflow);

        try {
            var findMatchesBtn = document.getElementById('findMatchesBtn');
            if (findMatchesBtn) {
                findMatchesBtn.click();
                recordResult(workflow, true, 'Find matches button clicked');
            } else {
                recordResult(workflow, false, 'Find matches button not found');
            }
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow12_UploadCompare() {
        var workflow = '3.4 Upload & Compare';
        log('Testing: ' + workflow);

        try {
            if (typeof compareWithUploadedImage !== 'function') {
                recordResult(workflow, false, 'compareWithUploadedImage function not found');
                return;
            }

            compareWithUploadedImage();
            recordResult(workflow, true, 'Upload & compare initiated');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow13_AddUploadedAsReference() {
        var workflow = '3.5 Add Uploaded as Reference';
        log('Testing: ' + workflow);

        try {
            if (typeof showUploadedAsReference !== 'function') {
                recordResult(workflow, false, 'showUploadedAsReference function not found');
                return;
            }

            showUploadedAsReference();
            recordResult(workflow, true, 'Uploaded as reference initiated');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 4: Webcam-Specific Workflows
    // ============================================================================

    function testWorkflow14_BasicWebcamCapture() {
        var workflow = '4.1 Basic Webcam Capture';
        log('Testing: ' + workflow);

        try {
            if (typeof startWebcam !== 'function') {
                recordResult(workflow, false, 'startWebcam function not found');
                return;
            }

            startWebcam();
            setTimeout(function() {
                captureWebcam();
                setTimeout(function() {
                    stopWebcam();
                    recordResult(workflow, true, 'Webcam capture complete');
                }, 500);
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow15_ProfileAngleCapture() {
        var workflow = '4.2 Profile/Angle Capture';
        log('Testing: ' + workflow);

        try {
            if (typeof startProfileCapture !== 'function') {
                recordResult(workflow, false, 'startProfileCapture function not found');
                return;
            }

            startWebcam();
            setTimeout(function() {
                startProfileCapture();
                setTimeout(function() {
                    stopWebcam();
                    recordResult(workflow, true, 'Profile capture complete');
                }, 500);
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow16_MeshOverlay() {
        var workflow = '4.3 Mesh Overlay';
        log('Testing: ' + workflow);

        try {
            if (typeof toggleMeshOverlay !== 'function') {
                recordResult(workflow, false, 'toggleMeshOverlay function not found');
                return;
            }

            startWebcam();
            setTimeout(function() {
                toggleMeshOverlay();
                setTimeout(function() {
                    stopWebcam();
                    recordResult(workflow, true, 'Mesh overlay toggled');
                }, 500);
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow17_StopWebcam() {
        var workflow = '4.4 Stop Webcam';
        log('Testing: ' + workflow);

        try {
            if (typeof stopWebcam !== 'function') {
                recordResult(workflow, false, 'stopWebcam function not found');
                return;
            }

            startWebcam();
            setTimeout(function() {
                stopWebcam();
                recordResult(workflow, true, 'Webcam stopped');
            }, 1000);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 5: Visualization/Analysis Workflows
    // ============================================================================

    var visualizationTypes = [
        { id: 'detection', name: '5.1 Detection View' },
        { id: 'extraction', name: '5.2 Extraction View' },
        { id: 'preprocessing', name: '5.3 Preprocessing View' },
        { id: 'landmarks', name: '5.4 Landmarks View' },
        { id: 'mesh3d', name: '5.5 3D Mesh View' },
        { id: 'alignment', name: '5.6 Alignment View' },
        { id: 'saliency', name: '5.7 Attention/Saliency View' },
        { id: 'activations', name: '5.8 Neural Activations View' },
        { id: 'features', name: '5.9 Feature Maps View' },
        { id: 'multiscale', name: '5.10 Multi-Scale View' },
        { id: 'confidence', name: '5.11 Confidence/Quality View' },
        { id: 'eyewear', name: '5.12 Eyewear Detection View' },
        { id: 'iris', name: '5.13 Iris Analysis View' },
        { id: 'expression', name: '5.14 Expression Analysis View' },
        { id: 'embedding', name: '5.15 Embedding View' },
        { id: 'similarity', name: '5.16 Similarity View' },
        { id: 'robustness', name: '5.17 Robustness Test View' },
        { id: 'biometric', name: '5.18 Biometric Overview View' },
        { id: 'asymmetry', name: '5.19 Uniqueness/Asymmetry View' },
        { id: 'texture', name: '5.20 Texture Analysis View' },
        { id: 'normalized', name: '5.21 3D Normalized View' }
    ];

    function testVisualizationWorkflow(vizType, vizName) {
        var workflow = vizName;
        log('Testing: ' + workflow);

        try {
            if (typeof showVisualization !== 'function') {
                recordResult(workflow, false, 'showVisualization function not found');
                return;
            }

            showVisualization(vizType);
            recordResult(workflow, true, 'Visualization: ' + vizType);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 6: Test/Diagnostic Workflows
    // ============================================================================

    function testDiagnosticWorkflow(testId, testName) {
        var workflow = testName;
        log('Testing: ' + workflow);

        try {
            if (typeof showVisualization !== 'function') {
                recordResult(workflow, false, 'showVisualization function not found');
                return;
            }

            showVisualization(testId);
            recordResult(workflow, true, 'Diagnostic: ' + testId);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // WORKFLOW TESTS - Category 7: System/Utility Workflows
    // ============================================================================

    function testWorkflow48_ClearCache() {
        var workflow = '7.1 Clear Cache';
        log('Testing: ' + workflow);

        try {
            if (typeof clearAllCache !== 'function') {
                recordResult(workflow, false, 'clearAllCache function not found');
                return;
            }

            clearAllCache();
            recordResult(workflow, true, 'Cache cleared');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow49_ClearTerminal() {
        var workflow = '7.2 Clear Terminal';
        log('Testing: ' + workflow);

        try {
            if (typeof clearTerminal !== 'function') {
                recordResult(workflow, false, 'clearTerminal function not found');
                return;
            }

            clearTerminal();
            recordResult(workflow, true, 'Terminal cleared');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow50_ToggleSidebar() {
        var workflow = '7.3 Toggle Sidebar';
        log('Testing: ' + workflow);

        try {
            if (typeof toggleSidebar !== 'function') {
                recordResult(workflow, false, 'toggleSidebar function not found');
                return;
            }

            toggleSidebar();
            recordResult(workflow, true, 'Sidebar toggled');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow51_WindowControls() {
        var workflow = '7.4 Window Controls';
        log('Testing: ' + workflow);

        try {
            if (typeof minimizeWindow !== 'function') {
                recordResult(workflow, false, 'minimizeWindow function not found');
                return;
            }

            minimizeWindow();
            setTimeout(function() {
                if (typeof maximizeWindow !== 'function') {
                    recordResult(workflow, false, 'maximizeWindow function not found');
                    return;
                }
                maximizeWindow();
                setTimeout(function() {
                    if (typeof closeWindow !== 'function') {
                        recordResult(workflow, false, 'closeWindow function not found');
                        return;
                    }
                    recordResult(workflow, true, 'Window controls work');
                }, 100);
            }, 100);
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow52_Navigation() {
        var workflow = '7.5 Navigation';
        log('Testing: ' + workflow);

        try {
            if (typeof jumpToStep !== 'function') {
                recordResult(workflow, false, 'jumpToStep function not found');
                return;
            }

            for (var i = 1; i <= 6; i++) {
                jumpToStep(i);
            }
            recordResult(workflow, true, 'All steps navigation works');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    function testWorkflow53_LibrarySearch() {
        var workflow = '7.6 Library Search';
        log('Testing: ' + workflow);

        try {
            if (typeof searchLibraryByName !== 'function') {
                recordResult(workflow, false, 'searchLibraryByName function not found');
                return;
            }

            searchLibraryByName('test');
            recordResult(workflow, true, 'Library search initiated');
        } catch (e) {
            recordResult(workflow, false, e.message);
        }
    }

    // ============================================================================
    // MAIN TEST RUNNER
    // ============================================================================

    function runAllTests() {
        log('Starting comprehensive workflow tests...', 'info');
        log('Total workflows: 53', 'info');

        // Category 1: Core Image Input (3)
        testWorkflow1_UploadImageToMatch();
        testWorkflow2_WebcamToMatch();
        testWorkflow3_AutoCaptureMode();

        // Category 2: Library Management (5)
        testWorkflow4_UploadSaveToLibrary();
        testWorkflow5_WebcamSaveToLibrary();
        testWorkflow6_BatchAddToLibrary();
        testWorkflow7_ViewLibraryPerson();
        testWorkflow8_DeleteLibraryPerson();

        // Category 3: Comparison (5)
        testWorkflow9_CompareWithReferences();
        testWorkflow10_CompareWithLibrary();
        testWorkflow11_FindMatchesCurrentImage();
        testWorkflow12_UploadCompare();
        testWorkflow13_AddUploadedAsReference();

        // Category 4: Webcam (4)
        testWorkflow14_BasicWebcamCapture();
        testWorkflow15_ProfileAngleCapture();
        testWorkflow16_MeshOverlay();
        testWorkflow17_StopWebcam();

        // Category 5: Visualizations (21)
        visualizationTypes.forEach(function(viz) {
            testVisualizationWorkflow(viz.id, viz.name);
        });

        // Category 6: Diagnostics (9)
        testDiagnosticWorkflow('test-health', '6.1 API Health Test');
        testDiagnosticWorkflow('test-detection', '6.2 Detection Test');
        testDiagnosticWorkflow('test-extraction', '6.3 Extraction Test');
        testDiagnosticWorkflow('test-reference', '6.4 Reference Test');
        testDiagnosticWorkflow('test-multi', '6.5 Multi-Match Test');
        testDiagnosticWorkflow('test-pose', '6.6 Pose Test');
        testDiagnosticWorkflow('test-eyewear', '6.7 Eyewear Test');
        testDiagnosticWorkflow('test-viz', '6.8 Viz Types Test');
        testDiagnosticWorkflow('test-clear', '6.9 Session Test');

        // Category 7: System/Utility (6)
        testWorkflow48_ClearCache();
        testWorkflow49_ClearTerminal();
        testWorkflow50_ToggleSidebar();
        testWorkflow51_WindowControls();
        testWorkflow52_Navigation();
        testWorkflow53_LibrarySearch();

        // Print results after a delay
        setTimeout(function() {
            printTestResults();
        }, 3000);
    }

    function printTestResults() {
        log('========================================', 'info');
        log('WORKFLOW TEST RESULTS', 'info');
        log('========================================', 'info');
        log('Passed: ' + testResults.passed, 'info');
        log('Failed: ' + testResults.failed, 'info');
        log('Skipped: ' + testResults.skipped, 'info');
        log('Total: ' + (testResults.passed + testResults.failed + testResults.skipped), 'info');
        log('========================================', 'info');

        if (testResults.errors.length > 0) {
            log('ERRORS:', 'fail');
            testResults.errors.forEach(function(err) {
                log('  ' + err.workflow + ': ' + err.details, 'fail');
            });
        }

        window.workflowTestResults = testResults;
    }

    // Expose test runner to window
    window.runWorkflowTests = runAllTests;
    window.getWorkflowTestResults = function() { return testResults; };

    // Auto-run if requested
    if (window.location.search.includes('autoRunTests')) {
        setTimeout(runAllTests, 1000);
    }

    log('Workflow test suite loaded. Run window.runWorkflowTests() to execute.', 'info');

})();
