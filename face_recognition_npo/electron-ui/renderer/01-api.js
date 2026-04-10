// ============================================================================
// FILE: 01-api.js
// PURPOSE: All HTTP communication with Flask backend
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. DEPENDENCIES & CONFIGURATION
    // ============================================================================

    var API_BASE = 'http://localhost:3000/api';

    var ENDPOINTS = {
        HEALTH: '/health',
        EMBEDDING_INFO: '/embedding-info',
        DIAGNOSTICS: '/diagnostics',
        DETECT: '/detect',
        EXTRACT: '/extract',
        ADD_REFERENCE: '/add-reference',
        REFERENCES: '/references',
        REMOVE_REFERENCE: function(id) { return '/references/' + id; },
        ADD_REFERENCE_POSE: function(id) { return '/add-reference-pose/' + id; },
        COMPARE: '/compare',
        COMPARE_LIBRARY: function(id) { return '/compare/library/' + encodeURIComponent(id); },
        QUALITY: '/quality',
        EYEWEAR: '/eyewear',
        LIBRARY: '/library',
        ADD_PERSON: '/library/person',
        LIBRARY_PERSON: function(id) { return '/library/person/' + encodeURIComponent(id); },
        LIBRARY_MATCH: '/library/match',
        CLEAR: '/clear',
        STATUS: '/status',
        VISUALIZATIONS: function(type) { return '/visualizations/' + type; },
        VISUALIZATIONS_REF: function(type, refId) { return '/visualizations/' + type + '/reference/' + refId; },
        VISUALIZATIONS_COMPARE_OVERLAY: function(refId) { return '/visualizations/compare-overlay/' + refId; },
        VISUALIZATIONS_COMPARE_DIFF: function(refId) { return '/visualizations/compare-diff/' + refId; },
        WEBCAM_AVAILABLE: '/webcam/available',
        WEBCAM_CAPTURE: '/webcam/capture',
        WEBCAM_DETECT: '/webcam/detect'
    };

    // ============================================================================
    // 2. HELPER FUNCTIONS
    // ============================================================================

    /**
     * Extract base64 from data URL
     * @param {string} dataUrl - Data URL (e.g., "data:image/jpeg;base64,...")
     * @returns {string} - Pure base64 string
     */
    function extractBase64(dataUrl) {
        if (!dataUrl) return null;
        if (dataUrl.startsWith('data:')) {
            var parts = dataUrl.split(',');
            return parts.length > 1 ? parts[1] : dataUrl;
        }
        return dataUrl;
    }

    /**
     * Build fetch options
     * @param {object} body - Request body
     * @returns {object} - Fetch options
     */
    function buildOptions(body) {
        var options = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        };
        if (body) {
            options.body = JSON.stringify(body);
        }
        return options;
    }

    /**
     * Handle fetch response
     * @param {Response} response - Fetch response
     * @returns {Promise<object>}
     */
    function handleResponse(response) {
        return response.json().then(function(data) {
            if (!response.ok) {
                return { success: false, error: data.error || 'HTTP ' + response.status };
            }
            return { success: true, data: data };
        }).catch(function(err) {
            return { success: false, error: err.message };
        });
    }

    // ============================================================================
    // 3. PUBLIC API - Health & Session
    // ============================================================================

    /**
     * Check if API server is running
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchHealth() {
        return fetch(API_BASE + ENDPOINTS.HEALTH)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get embedding info
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchEmbeddingInfo() {
        return fetch(API_BASE + ENDPOINTS.EMBEDDING_INFO)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get diagnostics
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchDiagnostics() {
        return fetch(API_BASE + ENDPOINTS.DIAGNOSTICS)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Clear session state on server
     * @returns {Promise<{success: boolean, error?: string}>}
     */
    function fetchClear() {
        return fetch(API_BASE + ENDPOINTS.CLEAR, { method: 'POST' })
            .then(function(response) {
                if (!response.ok) return { success: false, error: 'HTTP ' + response.status };
                return { success: true };
            })
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get server status
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchStatus() {
        return fetch(API_BASE + ENDPOINTS.STATUS)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 4. PUBLIC API - Detection & Extraction
    // ============================================================================

    /**
     * Detect faces in image
     * @param {string} imageData - Base64 or data URL image data
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchDetect(imageData) {
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);

        return fetch(API_BASE + ENDPOINTS.DETECT, buildOptions({ image: base64 }))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Extract embeddings from detected face
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchExtract() {
        return fetch(API_BASE + ENDPOINTS.EXTRACT, buildOptions())
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get quality metrics
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchQuality() {
        return fetch(API_BASE + ENDPOINTS.QUALITY)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Detect eyewear
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchEyewear() {
        return fetch(API_BASE + ENDPOINTS.EYEWEAR)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 5. PUBLIC API - References
    // ============================================================================

    /**
     * Add temporary reference for comparison
     * @param {string} imageData - Base64 or data URL image data
     * @param {string} name - Reference name
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchAddReference(imageData, name) {
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);
        var body = { image: base64 };
        if (name) body.name = name;

        return fetch(API_BASE + ENDPOINTS.ADD_REFERENCE, buildOptions(body))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Add pose variant to existing reference
     * @param {number} refId - Reference ID
     * @param {string} imageData - Base64 or data URL image data
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchAddReferencePose(refId, imageData) {
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);

        return fetch(API_BASE + ENDPOINTS.ADD_REFERENCE_POSE(refId), buildOptions({ image: base64 }))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get all temporary references
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchReferences() {
        return fetch(API_BASE + ENDPOINTS.REFERENCES)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Remove a temporary reference
     * @param {number} index - Reference index
     * @returns {Promise<{success: boolean, error?: string}>}
     */
    function fetchRemoveReference(index) {
        return fetch(API_BASE + ENDPOINTS.REMOVE_REFERENCE(index), { method: 'DELETE' })
            .then(function(response) {
                if (!response.ok) return { success: false, error: 'HTTP ' + response.status };
                return { success: true };
            })
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 6. PUBLIC API - Comparison
    // ============================================================================

    /**
     * Compare query against temporary references
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchCompare() {
        return fetch(API_BASE + ENDPOINTS.COMPARE, buildOptions())
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Compare query against specific library person
     * @param {string} personId - Library person ID
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchCompareLibrary(personId) {
        if (!personId) {
            return Promise.resolve({ success: false, error: 'Person ID required' });
        }

        return fetch(API_BASE + ENDPOINTS.COMPARE_LIBRARY(personId), buildOptions())
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Compare query against entire library
     * @param {string} imageData - Base64 or data URL image data
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchLibraryMatch(imageData) {
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);

        return fetch(API_BASE + ENDPOINTS.LIBRARY_MATCH, buildOptions({ image: base64 }))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 7. PUBLIC API - Library Management
    // ============================================================================

    /**
     * Get all library persons
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchLibrary() {
        return fetch(API_BASE + ENDPOINTS.LIBRARY)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Add new person to library
     * @param {string} name - Person name
     * @param {string} imageData - Base64 or data URL image data
     * @param {string} notes - Optional notes
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchAddPerson(name, imageData, notes) {
        if (!name) {
            return Promise.resolve({ success: false, error: 'Name required' });
        }
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);
        var body = { name: name, image: base64 };
        if (notes) body.notes = notes;

        return fetch(API_BASE + ENDPOINTS.ADD_PERSON, buildOptions(body))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get specific library person details
     * @param {string} personId - Library person ID
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchLibraryPerson(personId) {
        if (!personId) {
            return Promise.resolve({ success: false, error: 'Person ID required' });
        }

        return fetch(API_BASE + ENDPOINTS.LIBRARY_PERSON(personId))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Delete library person
     * @param {string} personId - Library person ID
     * @returns {Promise<{success: boolean, error?: string}>}
     */
    function fetchDeletePerson(personId) {
        if (!personId) {
            return Promise.resolve({ success: false, error: 'Person ID required' });
        }

        return fetch(API_BASE + ENDPOINTS.LIBRARY_PERSON(personId), { method: 'DELETE' })
            .then(function(response) {
                if (!response.ok) return { success: false, error: 'HTTP ' + response.status };
                return { success: true };
            })
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 8. PUBLIC API - Visualizations
    // ============================================================================

    /**
     * Get visualization for query image
     * @param {string} vizType - Visualization type
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchVisualization(vizType) {
        if (!vizType) {
            return Promise.resolve({ success: false, error: 'Visualization type required' });
        }

        return fetch(API_BASE + ENDPOINTS.VISUALIZATIONS(vizType))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get visualization for reference image
     * @param {string} vizType - Visualization type
     * @param {number} refId - Reference ID (integer)
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchReferenceVisualization(vizType, refId) {
        if (!vizType) {
            return Promise.resolve({ success: false, error: 'Visualization type required' });
        }
        if (typeof refId !== 'number') {
            return Promise.resolve({ success: false, error: 'Reference ID required' });
        }

        return fetch(API_BASE + ENDPOINTS.VISUALIZATIONS_REF(vizType, refId))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get compare overlay visualization
     * @param {number} refId - Reference ID
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchCompareOverlay(refId) {
        if (typeof refId !== 'number') {
            return Promise.resolve({ success: false, error: 'Reference ID required' });
        }

        return fetch(API_BASE + ENDPOINTS.VISUALIZATIONS_COMPARE_OVERLAY(refId))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Get compare diff visualization
     * @param {number} refId - Reference ID
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchCompareDiff(refId) {
        if (typeof refId !== 'number') {
            return Promise.resolve({ success: false, error: 'Reference ID required' });
        }

        return fetch(API_BASE + ENDPOINTS.VISUALIZATIONS_COMPARE_DIFF(refId))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 9. PUBLIC API - Webcam
    // ============================================================================

    /**
     * Check if webcam is available
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchWebcamAvailable() {
        return fetch(API_BASE + ENDPOINTS.WEBCAM_AVAILABLE)
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Capture webcam frame
     * @param {string} imageData - Base64 or data URL image data
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchWebcamCapture(imageData) {
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);

        return fetch(API_BASE + ENDPOINTS.WEBCAM_CAPTURE, buildOptions({ image: base64 }))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    /**
     * Detect faces from webcam
     * @param {string} imageData - Base64 or data URL image data
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function fetchWebcamDetect(imageData) {
        if (!imageData) {
            return Promise.resolve({ success: false, error: 'Image data required' });
        }

        var base64 = extractBase64(imageData);

        return fetch(API_BASE + ENDPOINTS.WEBCAM_DETECT, buildOptions({ image: base64 }))
            .then(handleResponse)
            .catch(function(err) {
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 10. EXPORTS
    // ============================================================================

    // Expose API functions to global scope
    global.API = {
        // Health & Session
        fetchHealth: fetchHealth,
        fetchEmbeddingInfo: fetchEmbeddingInfo,
        fetchDiagnostics: fetchDiagnostics,
        fetchClear: fetchClear,
        fetchStatus: fetchStatus,

        // Detection & Extraction
        fetchDetect: fetchDetect,
        fetchExtract: fetchExtract,
        fetchQuality: fetchQuality,
        fetchEyewear: fetchEyewear,

        // References
        fetchAddReference: fetchAddReference,
        fetchAddReferencePose: fetchAddReferencePose,
        fetchReferences: fetchReferences,
        fetchRemoveReference: fetchRemoveReference,

        // Comparison
        fetchCompare: fetchCompare,
        fetchCompareLibrary: fetchCompareLibrary,
        fetchLibraryMatch: fetchLibraryMatch,

        // Library
        fetchLibrary: fetchLibrary,
        fetchAddPerson: fetchAddPerson,
        fetchLibraryPerson: fetchLibraryPerson,
        fetchDeletePerson: fetchDeletePerson,

        // Visualizations
        fetchVisualization: fetchVisualization,
        fetchReferenceVisualization: fetchReferenceVisualization,
        fetchCompareOverlay: fetchCompareOverlay,
        fetchCompareDiff: fetchCompareDiff,

        // Webcam
        fetchWebcamAvailable: fetchWebcamAvailable,
        fetchWebcamCapture: fetchWebcamCapture,
        fetchWebcamDetect: fetchWebcamDetect,

        // Constants
        BASE: API_BASE,
        ENDPOINTS: ENDPOINTS
    };

})(window);
