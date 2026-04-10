// ============================================================================
// FILE: 06-viz.js
// PURPOSE: All visualization handling
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. CONFIGURATION
    // ============================================================================

    var DEFAULT_VIZ_TYPE = 'detection';

    var VALID_VIZ_TYPES = [
        'detection', 'extraction', 'preprocessing', 'landmarks', 'mesh3d',
        'alignment', 'saliency', 'activations', 'features', 'multiscale',
        'confidence', 'eyewear', 'iris', 'expression', 'embedding',
        'similarity', 'robustness', 'biometric', 'asymmetry', 'texture', 'normalized'
    ];

    // ============================================================================
    // 2. INTERNAL - Rendering
    // ============================================================================

    /**
     * Render visualization to content element
     * @param {string} visualizationData - Base64 or HTML visualization
     * @param {string} vizType - Visualization type
     * @param {object} data - Optional data object
     */
    function renderVisualization(visualizationData, vizType, data) {
        var content = document.getElementById('vizContent');
        
        if (!content || !visualizationData) {
            return;
        }

        // Check if it's a data table or image
        if (data && typeof data === 'object') {
            // Render as data table with image
            var html = '<div class="viz-image-container">';
            if (visualizationData.startsWith('data:')) {
                html += '<img src="' + visualizationData + '" alt="' + vizType + '">';
            } else {
                html += '<img src="data:image/png;base64,' + visualizationData + '" alt="' + vizType + '">';
            }
            html += '</div>';
            html += '<div class="viz-data-container">';
            html += UI.formatDataAsTable(data);
            html += '</div>';
            content.innerHTML = html;
        } else if (visualizationData.startsWith('data:image')) {
            // Render as image only
            content.innerHTML = '<div class="viz-image-container"><img src="' + visualizationData + '" alt="' + vizType + '"></div>';
        } else if (visualizationData.startsWith('data:')) {
            // Render as is (might be SVG or HTML)
            content.innerHTML = '<div class="viz-image-container">' + visualizationData + '</div>';
        } else {
            // Assume base64
            content.innerHTML = '<div class="viz-image-container"><img src="data:image/png;base64,' + visualizationData + '" alt="' + vizType + '"></div>';
        }
    }

    /**
     * Show visualization placeholder
     */
    function showVisualizationPlaceholder() {
        var content = document.getElementById('vizContent');
        
        if (content) {
            content.innerHTML = '<div class="viz-placeholder"><p>Run analysis to see visualizations</p></div>';
        }
    }

    // ============================================================================
    // 3. PUBLIC API - Visualization Display
    // ============================================================================

    /**
     * Show visualization for specified type
     * @param {string} vizType - Visualization type
     */
    function showVisualization(vizType) {
        var content = document.getElementById('vizContent');
        
        if (!content) {
            console.error('[VIZ] vizContent element not found');
            return;
        }
        
        // Validate viz type
        if (!vizType || VALID_VIZ_TYPES.indexOf(vizType) === -1) {
            vizType = DEFAULT_VIZ_TYPE;
        }
        
        // Check prerequisites
        if (!State.hasDetectedFaces()) {
            content.innerHTML = '<div class="viz-placeholder">' +
                '<p class="text-error-inline">No face detected</p>' +
                '<p>1. Upload an image</p>' +
                '<p>2. Click "Find Faces"</p>' +
                '<p>3. Click "Create Signature"</p>' +
                '<p>4. Then click visualization tabs</p>' +
                '</div>';
            return;
        }
        
        if (!State.hasQueryEmbedding()) {
            content.innerHTML = '<div class="viz-placeholder">' +
                '<p class="text-error-inline">No embedding extracted</p>' +
                '<p>Click "Create Signature" first</p>' +
                '</div>';
            return;
        }
        
        // Check cache first
        var cached = State.getVisualization(vizType);
        if (cached) {
            renderVisualization(cached, vizType, State.getVisualization(vizType + '_data'));
            return;
        }
        
        // Fetch from API
        UI.logToTerminal('> Loading visualization: ' + vizType, 'info');
        
        API.fetchVisualization(vizType)
            .then(function(result) {
                if (result.success && result.data && result.data.visualization) {
                    // Cache the data
                    State.addVisualizationData(vizType, result.data.visualization);
                    
                    // Store additional data if present
                    if (result.data.data) {
                        State.addVisualizationData(vizType + '_data', result.data.data);
                    }
                    
                    renderVisualization(result.data.visualization, vizType, result.data.data);
                } else {
                    showVisualizationPlaceholder();
                }
            })
            .catch(function(err) {
                console.error('[VIZ] Error:', err);
                showVisualizationPlaceholder();
            });
    }

    /**
     * Show visualization placeholder
     */
    function showVisualizationPlaceholder() {
        var content = document.getElementById('vizContent');
        
        if (content) {
            content.innerHTML = '<div class="viz-placeholder"><p>Run analysis to see visualizations</p></div>';
        }
    }

    // ============================================================================
    // 4. PUBLIC API - Reference Visualizations
    // ============================================================================

    /**
     * Show visualization for reference image
     * @param {number} refIndex - Reference index (integer)
     * @param {string} vizType - Visualization type
     */
    function showReferenceVisualization(refIndex, vizType) {
        if (typeof refIndex !== 'number') {
            console.error('[VIZ] refIndex must be a number, got:', refIndex);
            return;
        }

        API.fetchReferenceVisualization(vizType, refIndex)
            .then(function(result) {
                if (result.success && result.data && result.data.visualization) {
                    var content = document.getElementById('refVizContent');
                    
                    if (content) {
                        renderVisualization(result.data.visualization, vizType, result.data.data);
                    }
                }
            })
            .catch(function(err) {
                console.error('[VIZ] Reference visualization error:', err);
            });
    }

    // ============================================================================
    // 5. PUBLIC API - Tab Management
    // ============================================================================

    /**
     * Handle viz tab click
     * @param {string} vizType - Visualization type
     */
    function handleVizTabClick(vizType) {
        // Update active tab
        var tabs = document.querySelectorAll('.viz-tab');
        tabs.forEach(function(tab) {
            if (tab.dataset.viz === vizType) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Show visualization
        showVisualization(vizType);
    }

    // ============================================================================
    // 6. EXPORTS
    // ============================================================================

    global.Viz = {
        showVisualization: showVisualization,
        showVisualizationPlaceholder: showVisualizationPlaceholder,
        showReferenceVisualization: showReferenceVisualization,
        handleVizTabClick: handleVizTabClick,
        VALID_VIZ_TYPES: VALID_VIZ_TYPES
    };

})(window);
