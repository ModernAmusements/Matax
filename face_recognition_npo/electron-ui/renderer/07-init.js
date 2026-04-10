// ============================================================================
// FILE: 07-init.js
// PURPOSE: Bootstrap and event listeners
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. INITIALIZATION
    // ============================================================================

    /**
     * Initialize the application
     */
    function init() {
        console.log('[INIT] Starting Face Recognition App...');
        
        // Setup event listeners
        setupEventListeners();
        
        // Initialize terminal
        initTerminal();
        
        // Check API connection
        checkAPI();
        
        // Load existing references
        loadReferences();
        
        // Convert viz tabs to radio buttons
        convertVizTabsToRadio();
        
        console.log('[INIT] Application initialized');
    }

    // ============================================================================
    // 2. EVENT LISTENERS
    // ============================================================================

    /**
     * Setup all event listeners
     */
    function setupEventListeners() {
        // File inputs
        var imageInput = document.getElementById('imageInput');
        if (imageInput) {
            imageInput.addEventListener('change', Workflows.handleImageSelect);
        }

        var refInput = document.getElementById('refInput');
        if (refInput) {
            refInput.addEventListener('change', Workflows.handleReferenceSelect);
        }

        var libraryUpload = document.getElementById('libraryUpload');
        if (libraryUpload) {
            libraryUpload.addEventListener('change', Workflows.handleLibraryUpload);
        }

        // Buttons
        setupButtonListeners();
        
        // Viz tabs
        setupVizTabs();
        
        // Sidebar
        setupSidebar();
    }

    /**
     * Setup button event listeners
     */
    function setupButtonListeners() {
        // Detect button
        var detectBtn = document.getElementById('detectBtn');
        if (detectBtn) {
            detectBtn.addEventListener('click', function() {
                Workflows.detectFaces();
            });
        }

        // Extract button
        var extractBtn = document.getElementById('extractBtn');
        if (extractBtn) {
            extractBtn.addEventListener('click', function() {
                Workflows.extractFeatures();
            });
        }

        // Compare button
        var compareBtn = document.getElementById('compareBtn');
        if (compareBtn) {
            compareBtn.addEventListener('click', function() {
                Compare.compareFaces();
            });
        }

        // Compare with library button
        var compareLibraryBtn = document.getElementById('compareLibraryBtn');
        if (compareLibraryBtn) {
            compareLibraryBtn.addEventListener('click', function() {
                Compare.compareWithLibrary();
            });
        }

        // Clear cache button
        var clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', function() {
                Workflows.clearAllCache();
            });
        }

        // Library save button
        var saveLibraryBtn = document.getElementById('saveLibraryBtn');
        if (saveLibraryBtn) {
            saveLibraryBtn.addEventListener('click', function() {
                var nameInput = document.getElementById('libraryPersonName');
                var notesInput = document.getElementById('libraryPersonNotes');
                var name = nameInput ? nameInput.value.trim() : '';
                var notes = notesInput ? notesInput.value.trim() : '';
                Workflows.saveToLibrary(name, notes);
            });
        }

        // Library cancel button
        var cancelLibraryBtn = document.getElementById('cancelLibraryBtn');
        if (cancelLibraryBtn) {
            cancelLibraryBtn.addEventListener('click', function() {
                Workflows.closeLibraryModal();
            });
        }
    }

    /**
     * Setup visualization tabs
     */
    function setupVizTabs() {
        var tabs = document.querySelectorAll('.viz-tab');
        
        tabs.forEach(function(tab) {
            tab.addEventListener('click', function() {
                var vizType = this.dataset.viz;
                if (vizType) {
                    Viz.handleVizTabClick(vizType);
                }
            });
        });
    }

    /**
     * Setup sidebar
     */
    function setupSidebar() {
        var sidebarToggle = document.getElementById('sidebarToggle');
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', function() {
                toggleSidebar();
            });
        }

        var terminalToggle = document.getElementById('terminalToggle');
        if (terminalToggle) {
            terminalToggle.addEventListener('click', function() {
                UI.toggleTerminal();
            });
        }
    }

    // ============================================================================
    // 3. HELPERS
    // ============================================================================

    /**
     * Check API connection
     */
    function checkAPI() {
        API.fetchHealth()
            .then(function(result) {
                if (result.success) {
                    UI.logToTerminal('> API connected', 'success');
                } else {
                    UI.logToTerminal('> API error: ' + result.error, 'error');
                }
            })
            .catch(function(err) {
                UI.logToTerminal('> API unreachable: ' + err.message, 'error');
            });
    }

    /**
     * Load existing references from server
     */
    function loadReferences() {
        API.fetchReferences()
            .then(function(result) {
                if (result.success && result.data && result.data.references) {
                    result.data.references.forEach(function(ref) {
                        State.addReference(ref);
                    });
                    UI.logToTerminal('> Loaded ' + result.data.count + ' reference(s)', 'info');
                }
            })
            .catch(function(err) {
                UI.logToTerminal('> Failed to load references: ' + err.message, 'error');
            });
    }

    /**
     * Initialize terminal
     */
    function initTerminal() {
        var content = document.getElementById('terminalLogContent');
        if (content) {
            content.innerHTML = '';
            var welcome = document.createElement('div');
            welcome.className = 'terminal-line terminal-info';
            var timestamp = new Date().toLocaleTimeString();
            welcome.textContent = timestamp + ' Face Recognition System v2.0';
            content.appendChild(welcome);
        }
    }

    /**
     * Convert viz tabs to radio-style behavior
     */
    function convertVizTabsToRadio() {
        var vizTabs = document.getElementById('vizTabs');
        if (!vizTabs || vizTabs.querySelector('.viz-input')) return;

        var tabs = vizTabs.querySelectorAll('.viz-tab');
        
        tabs.forEach(function(tab) {
            tab.addEventListener('click', function() {
                tabs.forEach(function(t) { t.classList.remove('active'); });
                this.classList.add('active');
            });
        });
    }

    /**
     * Toggle sidebar
     */
    function toggleSidebar() {
        var sidebar = document.getElementById('sidebar');
        
        if (sidebar) {
            if (State.isSidebarOpen()) {
                sidebar.classList.remove('active');
                State.setSidebarOpen(false);
            } else {
                sidebar.classList.add('active');
                State.setSidebarOpen(true);
            }
        }
    }

    // ============================================================================
    // 4. EXPOSE TO WINDOW
    // ============================================================================

    // Expose init to window for DOMContentLoaded
    global.AppInit = init;

    // Also expose individual functions that need to be globally accessible
    global.handleImageSelect = Workflows.handleImageSelect;
    global.handleReferenceSelect = Workflows.handleReferenceSelect;
    global.handleLibraryUpload = Workflows.handleLibraryUpload;
    global.detectFaces = Workflows.detectFaces;
    global.extractFeatures = Workflows.extractFeatures;
    global.compareFaces = Compare.compareFaces;
    global.compareWithLibrary = Compare.compareWithLibrary;
    global.clearAllCache = Workflows.clearAllCache;
    global.removeReference = Workflows.removeReference;
    global.showVisualization = Viz.showVisualization;
    global.showToast = UI.showToast;
    global.logToTerminal = UI.logToTerminal;
    global.toggleTerminal = UI.toggleTerminal;
    global.saveToLibrary = Workflows.saveToLibrary;
    global.loadLibrary = Workflows.loadLibrary;
    global.toggleSidebar = toggleSidebar;

    // ============================================================================
    // 5. AUTO-INIT
    // ============================================================================

    // Run init when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})(window);
