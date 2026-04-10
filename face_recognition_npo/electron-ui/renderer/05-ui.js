// ============================================================================
// FILE: 05-ui.js
// PURPOSE: All UI utilities - Toast, Loading, Terminal, DOM updates
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. CONFIGURATION
    // ============================================================================

    var TOAST_DURATION = 5000;
    var TOAST_TRANSITION = 300;

    var TOAST_TYPES = {
        INFO: 'info',
        SUCCESS: 'success',
        WARNING: 'warning',
        ERROR: 'error'
    };

    var STATUS_CLASSES = {
        INFO: 'status-info',
        SUCCESS: 'status-success',
        WARNING: 'status-warning',
        ERROR: 'status-error'
    };

    // ============================================================================
    // 2. PUBLIC API - Toast Notifications
    // ============================================================================

    /**
     * Show toast notification
     * @param {string} message - Toast message
     * @param {string} type - Toast type (info|success|warning|error)
     */
    function showToast(message, type) {
        type = type || TOAST_TYPES.INFO;
        
        var container = document.getElementById('toastContainer');
        
        if (!container) {
            console.log('[' + type + '] ' + message);
            return;
        }
        
        var toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.innerHTML = '<span>' + message + '</span>';
        container.appendChild(toast);
        
        setTimeout(function() {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            setTimeout(function() {
                toast.remove();
            }, TOAST_TRANSITION);
        }, TOAST_DURATION);
    }

    // ============================================================================
    // 3. PUBLIC API - Loading Overlay
    // ============================================================================

    /**
     * Show loading overlay
     * @param {string} text - Loading text
     */
    function showLoading(text) {
        text = text || 'Loading...';
        
        var loadingText = document.getElementById('loadingText');
        var overlay = document.getElementById('loadingOverlay');
        
        if (loadingText) {
            loadingText.textContent = text;
        }
        
        if (overlay) {
            overlay.classList.add('active');
        }
    }

    /**
     * Hide loading overlay
     */
    function hideLoading() {
        var overlay = document.getElementById('loadingOverlay');
        
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    // ============================================================================
    // 4. PUBLIC API - Terminal
    // ============================================================================

    /**
     * Log message to terminal
     * @param {string} message - Log message
     * @param {string} type - Log type (info|success|warning|error|command)
     */
    function logToTerminal(message, type) {
        type = type || 'info';
        
        var content = document.getElementById('terminalLogContent');
        if (!content) {
            console.log('[' + type + '] ' + message);
            return;
        }

        var entry = document.createElement('div');
        entry.className = 'terminal-line terminal-' + type;

        var timestamp = new Date().toLocaleTimeString();
        entry.textContent = timestamp + ' ' + message;

        content.appendChild(entry);
        content.scrollTop = content.scrollHeight;
    }

    /**
     * Clear terminal output
     */
    function clearTerminal() {
        var content = document.getElementById('terminalLogContent');
        
        if (content) {
            content.innerHTML = '';
        }
    }

    /**
     * Toggle terminal visibility
     */
    function toggleTerminal() {
        var terminal = document.getElementById('terminalContainer');
        
        if (terminal) {
            if (State.isTerminalExpanded()) {
                terminal.classList.remove('active');
                State.setTerminalExpanded(false);
            } else {
                terminal.classList.add('active');
                State.setTerminalExpanded(true);
            }
        }
    }

    // ============================================================================
    // 5. PUBLIC API - Status Updates
    // ============================================================================

    /**
     * Update status element
     * @param {string} elementId - Element ID
     * @param {string} message - Status message
     * @param {string} type - Status type
     */
    function updateStatus(elementId, message, type) {
        type = type || TOAST_TYPES.INFO;
        
        var element = document.getElementById(elementId);
        
        if (element) {
            element.textContent = message;
            element.className = 'status ' + (STATUS_CLASSES[type] || '');
        }
    }

    /**
     * Enable/disable button
     * @param {string} buttonId - Button ID
     * @param {boolean} enabled - Enabled state
     */
    function setButtonEnabled(buttonId, enabled) {
        var button = document.getElementById(buttonId);
        
        if (button) {
            button.disabled = !enabled;
        }
    }

    /**
     * Update button text
     * @param {string} buttonId - Button ID
     * @param {string} text - New button text
     */
    function setButtonText(buttonId, text) {
        var button = document.getElementById(buttonId);
        
        if (button) {
            button.textContent = text;
        }
    }

    // ============================================================================
    // 6. PUBLIC API - Scores Dropdown
    // ============================================================================

    /**
     * Expand scores dropdown
     */
    function expandScoresDropdown() {
        var toggle = document.querySelector('.scores-toggle');
        var scores = document.querySelector('.comparison-scores');
        
        if (toggle && scores) {
            toggle.setAttribute('data-expanded', 'true');
            scores.setAttribute('data-visible', 'true');
            var icon = toggle.querySelector('.toggle-icon');
            if (icon) icon.textContent = '▼';
        }
    }

    /**
     * Collapse scores dropdown
     */
    function collapseScoresDropdown() {
        var toggle = document.querySelector('.scores-toggle');
        var scores = document.querySelector('.comparison-scores');
        
        if (toggle && scores) {
            toggle.setAttribute('data-expanded', 'false');
            scores.setAttribute('data-visible', 'false');
            var icon = toggle.querySelector('.toggle-icon');
            if (icon) icon.textContent = '▶';
        }
    }

    // ============================================================================
    // 7. PUBLIC API - Image Display
    // ============================================================================

    /**
     * Set image source
     * @param {string} elementId - Image element ID
     * @param {string} imageData - Base64 image data
     */
    function setImageSrc(elementId, imageData) {
        var element = document.getElementById(elementId);
        
        if (element && imageData) {
            element.src = imageData;
            element.style.display = 'inline-block';
        }
    }

    /**
     * Clear image
     * @param {string} elementId - Image element ID
     */
    function clearImage(elementId) {
        var element = document.getElementById(elementId);
        
        if (element) {
            element.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
            element.style.display = 'none';
        }
    }

    // ============================================================================
    // 8. PUBLIC API - DOM Helpers
    // ============================================================================

    /**
     * Show element
     * @param {string} elementId - Element ID
     */
    function showElement(elementId) {
        var element = document.getElementById(elementId);
        
        if (element) {
            element.classList.remove('hidden');
            element.classList.add('visible');
        }
    }

    /**
     * Hide element
     * @param {string} elementId - Element ID
     */
    function hideElement(elementId) {
        var element = document.getElementById(elementId);
        
        if (element) {
            element.classList.add('hidden');
            element.classList.remove('visible');
        }
    }

    /**
     * Set element text content
     * @param {string} elementId - Element ID
     * @param {string} text - Text content
     */
    function setElementText(elementId, text) {
        var element = document.getElementById(elementId);
        
        if (element) {
            element.textContent = text;
        }
    }

    // ============================================================================
    // 9. PUBLIC API - Data Formatting
    // ============================================================================

    /**
     * Format key for display
     * @param {string} key - Key to format
     * @returns {string}
     */
    function formatKey(key) {
        return key.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim();
    }

    /**
     * Format data as table HTML
     * @param {object} data - Data object
     * @returns {string}
     */
    function formatDataAsTable(data) {
        if (!data || typeof data !== 'object') {
            return '<p>No data</p>';
        }
        
        var html = '<table class="data-table">';
        
        for (var key in data) {
            if (data.hasOwnProperty(key)) {
                var formattedKey = formatKey(key);
                var value = data[key];
                var displayValue = value;
                
                if (typeof value === 'number') {
                    displayValue = value.toFixed(4);
                } else if (typeof value === 'object') {
                    displayValue = JSON.stringify(value);
                }
                
                html += '<tr><td>' + formattedKey + '</td><td>' + displayValue + '</td></tr>';
            }
        }
        
        html += '</table>';
        return html;
    }

    /**
     * Get score class based on value
     * @param {number} score - Score value (0-1)
     * @returns {string}
     */
    function getScoreClass(score) {
        if (score >= 0.75) return 'score-high';
        if (score >= 0.55) return 'score-medium';
        return 'score-low';
    }

    /**
     * Get match label based on score
     * @param {number} score - Score value (0-1)
     * @returns {string}
     */
    function getMatchLabel(score) {
        if (score >= 0.75) return 'Full Match';
        if (score >= 0.55) return 'Possible Match';
        return 'No Match';
    }

    // ============================================================================
    // 10. EXPORTS
    // ============================================================================

    global.UI = {
        // Toast
        showToast: showToast,

        // Loading
        showLoading: showLoading,
        hideLoading: hideLoading,

        // Terminal
        logToTerminal: logToTerminal,
        clearTerminal: clearTerminal,
        toggleTerminal: toggleTerminal,

        // Status
        updateStatus: updateStatus,
        setButtonEnabled: setButtonEnabled,
        setButtonText: setButtonText,

        // Scores
        expandScoresDropdown: expandScoresDropdown,
        collapseScoresDropdown: collapseScoresDropdown,

        // Images
        setImageSrc: setImageSrc,
        clearImage: clearImage,

        // DOM
        showElement: showElement,
        hideElement: hideElement,
        setElementText: setElementText,

        // Formatting
        formatKey: formatKey,
        formatDataAsTable: formatDataAsTable,
        getScoreClass: getScoreClass,
        getMatchLabel: getMatchLabel
    };

})(window);
