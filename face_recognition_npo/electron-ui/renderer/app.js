/**
 * Face Recognition App - Renderer JavaScript
 * Communicates with Python Flask API
 */

const API_BASE = 'http://localhost:3000/api';

// State
let currentImage = null;
let currentFaceThumbnails = [];
let currentQueryEmbedding = null;
let references = [];
let selectedReferenceId = null;
let visualizationData = {};
let terminalExpanded = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPI();
    initTerminal();
});

function setupEventListeners() {
    // Convert viz-tabs to radio buttons if not already done
    const vizTabs = document.getElementById('vizTabs');
    if (vizTabs && !vizTabs.querySelector('.viz-input')) {
        convertVizTabsToRadio();
    }
    
    // Handle viz-tab clicks (for buttons/labels)
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const vizType = e.target.dataset.viz || e.target.getAttribute('for')?.replace('viz-', '');
            if (vizType) {
                document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.viz-input').forEach(r => r.checked = false);
                e.target.classList.add('active');
                const radio = document.querySelector(`.viz-input[data-viz="${vizType}"]`);
                if (radio) radio.checked = true;
                
                // Update indicator position
                const currentVizTabs = document.getElementById('vizTabs');
                setTimeout(() => updateVizIndicatorPosition(currentVizTabs, vizType), 0);
                
                logToTerminal(`>>> CLICKED TAB: ${vizType}`, 'info');
                showVisualization(vizType);
            }
        });
    });
}

function convertVizTabsToRadio() {
    const vizTabs = document.getElementById('vizTabs');
    if (!vizTabs) return;
    
    const buttons = vizTabs.querySelectorAll('.viz-tab');
    if (buttons.length === 0) return;
    
    // Get current active tab
    const activeButton = vizTabs.querySelector('.viz-tab.active');
    const activeViz = activeButton ? activeButton.dataset.viz : 'detection';
    
    // Create radio inputs
    const radioHtml = Array.from(buttons).map((btn, i) => {
        const vizType = btn.dataset.viz;
        return `<input type="radio" name="viz" id="viz-${vizType}" class="viz-input" data-viz="${vizType}" ${vizType === activeViz ? 'checked' : ''}>`;
    }).join('');
    
    // Update buttons to labels
    const labelHtml = Array.from(buttons).map(btn => {
        const vizType = btn.dataset.viz;
        const label = btn.textContent;
        const isActive = btn.classList.contains('active');
        return `<label class="viz-tab ${isActive ? 'active' : ''}" for="viz-${vizType}" data-viz="${vizType}">${label}</label>`;
    }).join('');
    
    vizTabs.innerHTML = radioHtml + labelHtml;
    
    // Add change listeners
    const radios = vizTabs.querySelectorAll('.viz-input');
    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            const vizType = radio.dataset.viz;
            const currentVizTabs = document.getElementById('vizTabs');
            document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.viz-tab[data-viz="${vizType}"]`)?.classList.add('active');
            
            // Update indicator position
            setTimeout(() => updateVizIndicatorPosition(currentVizTabs, vizType), 0);
            
            logToTerminal(`>>> SELECTED TAB: ${vizType}`, 'info');
            showVisualization(vizType);
        });
    });
    
    // Track previous for directional animation
    setTimeout(() => trackPrevious(vizTabs), 0);
    
    // Update indicator position
    setTimeout(() => updateVizIndicatorPosition(vizTabs, activeViz), 0);
}

function updateVizIndicatorPosition(tabsEl, activeViz) {
    const activeTab = tabsEl.querySelector(`.viz-tab[data-viz="${activeViz}"]`);
    if (!activeTab) return;
    
    const containerRect = tabsEl.getBoundingClientRect();
    const tabRect = activeTab.getBoundingClientRect();
    const relativeLeft = tabRect.left - containerRect.left;
    
    tabsEl.style.setProperty('--indicator-left', `${relativeLeft}px`);
    tabsEl.style.setProperty('--indicator-width', `${tabRect.width}px`);
}

async function checkAPI() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        if (data.status === 'ok') {
            logToTerminal('> API connected', 'success');
            loadReferences();
        }
    } catch (err) {
        logToTerminal('> Cannot connect to API server', 'error');
        showToast('Cannot connect to API server. Make sure api_server.py is running.', 'error');
    }
}

async function loadReferences() {
    try {
        const response = await fetch(`${API_BASE}/references`);
        const data = await response.json();
        if (data.references) {
            references = data.references;
            updateReferenceList();
            logToTerminal(`> Loaded ${data.count} reference(s) from storage`, 'info');
        }
    } catch (err) {
        logToTerminal(`> Failed to load references: ${err.message}`, 'error');
    }
}

// Terminal Log Functions
function initTerminal() {
    const content = document.getElementById('terminalLogContent');
    if (content) {
        content.innerHTML = '';
        const welcome = document.createElement('div');
        welcome.className = 'terminal-line info';
        welcome.innerHTML = `<span class="timestamp">[${new Date().toLocaleTimeString('en-US', { hour12: false })}]</span> Face Recognition System v1.0`;
        content.appendChild(welcome);
    }
}

function logToTerminal(message, type = 'info') {
    const content = document.getElementById('terminalLogContent');
    if (!content) return;
    
    const line = document.createElement('div');
    line.className = `terminal-line ${type}`;
    
    const now = new Date();
    const timestamp = now.toLocaleTimeString('en-US', { hour12: false });
    
    line.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    content.appendChild(line);
    
    // Keep max 50 lines
    while (content.children.length > 50) {
        content.removeChild(content.firstChild);
    }
}

function toggleTerminal() {
    const terminalLog = document.getElementById('terminalLog');
    const terminalToggle = document.getElementById('terminalToggle');
    
    terminalExpanded = !terminalExpanded;
    
    if (terminalExpanded) {
        terminalLog.classList.add('expanded');
        terminalToggle.textContent = '[-]';
    } else {
        terminalLog.classList.remove('expanded');
        terminalToggle.textContent = '[+]';
    }
}

function clearTerminal() {
    const content = document.getElementById('terminalLogContent');
    if (content) {
        content.innerHTML = '';
    }
}

async function clearAllCache() {
    logToTerminal('> Clearing all cache...', 'info');
    
    const buttons = ['detectBtn', 'extractBtn', 'compareBtn'];
    buttons.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = true;
    });
    
    try {
        const response = await fetch(`${API_BASE}/clear`, { method: 'POST' });
        const data = await response.json();
        logToTerminal('> Backend cache cleared', 'success');
    } catch (err) {
        logToTerminal(`> Warning: Backend clear failed: ${err.message}`, 'warning');
    }
    
    currentImage = null;
    currentFaceThumbnails = [];
    currentQueryEmbedding = null;
    references = [];
    visualizationData = {};
    
    const selectedImageEl = document.getElementById('selectedImage');
    if (selectedImageEl) selectedImageEl.src = '';
    const previewContainerEl = document.getElementById('previewContainer');
    if (previewContainerEl) {
        previewContainerEl.classList.remove('visible');
        previewContainerEl.classList.add('hidden');
    }
    const step1ButtonsInitialEl = document.getElementById('step1ButtonsInitial');
    if (step1ButtonsInitialEl) step1ButtonsInitialEl.classList.remove('hidden');
    const step1ButtonsAfterEl = document.getElementById('step1ButtonsAfter');
    if (step1ButtonsAfterEl) step1ButtonsAfterEl.classList.add('hidden');
    const step1El = document.getElementById('step1');
    if (step1El) step1El.classList.remove('step-complete');
    const step2El = document.getElementById('step2');
    if (step2El) step2El.classList.remove('step-complete');
    const step3El = document.getElementById('step3');
    if (step3El) step3El.classList.remove('step-complete');
    const step4El = document.getElementById('step4');
    if (step4El) step4El.classList.remove('step-complete');
    const webcamStepEl = document.getElementById('webcamStep');
    if (webcamStepEl) webcamStepEl.classList.remove('step-complete');
    
    const detectBtn = document.getElementById('detectBtn');
    const extractBtn = document.getElementById('extractBtn');
    if (detectBtn) {
        detectBtn.classList.remove('btn-success');
        detectBtn.classList.add('btn-primary');
    }
    if (extractBtn) {
        extractBtn.classList.remove('btn-success');
        extractBtn.classList.add('btn-primary');
    }
    
    // Clear Step 5 preview
    const step5PreviewEl = document.getElementById('step5PreviewContainer');
    if (step5PreviewEl) {
        step5PreviewEl.classList.add('hidden');
        step5PreviewEl.classList.remove('visible');
    }
    const step5ImageEl = document.getElementById('step5SelectedImage');
    if (step5ImageEl) step5ImageEl.src = '';
    
    // Clear Step 5 faces
    const step5FacesEl = document.getElementById('step5FacesContainer');
    if (step5FacesEl) {
        step5FacesEl.classList.add('hidden');
        step5FacesEl.classList.remove('visible');
    }
    const step5GalleryEl = document.getElementById('step5FacesGallery');
    if (step5GalleryEl) step5GalleryEl.innerHTML = '';
    
    // Clear Step 5 status
    const step5StatusEl = document.getElementById('step5Status');
    if (step5StatusEl) {
        step5StatusEl.textContent = 'Upload an image to analyze';
        step5StatusEl.className = 'status';
    }
    
    // Clear currently uploaded section
    updateCurrentlyUploaded();
    
    showToast('Cache cleared', 'success');
}

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showToast('Please select an image file (JPEG, PNG, GIF, WebP)', 'error');
        logToTerminal(`> Invalid file type: ${file.type}`, 'error');
        event.target.value = '';
        return;
    }
    
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('Image too large (max 10MB)', 'warning');
        logToTerminal(`> File too large: ${(file.size / 1024 / 1024).toFixed(2)}MB`, 'warning');
        event.target.value = '';
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        // Fire-and-forget clear - don't await
        fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
            console.log('Clear failed:', err.message);
        });
        
        currentImage = e.target.result;
        
        // Update preview - with null checks
        const selectedImageEl = document.getElementById('selectedImage');
        if (selectedImageEl) selectedImageEl.src = currentImage;
        const previewContainerEl = document.getElementById('previewContainer');
        if (previewContainerEl) {
            previewContainerEl.classList.add('visible');
            previewContainerEl.classList.remove('hidden');
        }
        
        // Also show in Step 5 - with null checks
        const step5SelectedImageEl = document.getElementById('step5SelectedImage');
        if (step5SelectedImageEl) step5SelectedImageEl.src = currentImage;
        const step5PreviewContainerEl = document.getElementById('step5PreviewContainer');
        if (step5PreviewContainerEl) {
            step5PreviewContainerEl.classList.remove('hidden');
            step5PreviewContainerEl.classList.add('visible');
        }
        
        // Update status (handle if element doesn't exist)
        const detectStatusEl = document.getElementById('detectStatus');
        if (detectStatusEl) {
            detectStatusEl.textContent = 'Ready - auto-detecting...';
            detectStatusEl.className = 'status status-info';
        }
        
        const step5StatusEl = document.getElementById('step5Status');
        if (step5StatusEl) {
            step5StatusEl.textContent = 'Ready - auto-detecting...';
            step5StatusEl.className = 'status status-info';
        }
        
        resetSteps();
        markStepComplete('step1');
        event.target.value = '';
        
        // Update UI sections
        checkLibraryCompareButton();
        checkFindMatchesButton();
        updateCurrentlyUploaded();
        
        // Show success notification
        showToast('Image uploaded successfully', 'success');
        
        // Show after-upload buttons - with null checks
        const step1ButtonsInitialEl = document.getElementById('step1ButtonsInitial');
        if (step1ButtonsInitialEl) step1ButtonsInitialEl.classList.add('hidden');
        const step1ButtonsAfterEl = document.getElementById('step1ButtonsAfter');
        if (step1ButtonsAfterEl) step1ButtonsAfterEl.classList.remove('hidden');
        
        // Auto-run detection after upload (Option A)
        detectFaces().then(() => {
            // Auto-run extraction after detection
            extractFeatures().then(() => {
                // Scroll to compare section after extraction completes
                scrollToSection('step4');
            });
        });
    };
    reader.onerror = (err) => {
        logToTerminal('> Error reading file', 'error');
        showToast('Error reading file', 'error');
    };
    reader.readAsDataURL(file);
}

function resetSteps() {
    currentFaceThumbnails = [];
    currentQueryEmbedding = null;
    
    const facesContainerEl = document.getElementById('facesContainer');
    if (facesContainerEl) facesContainerEl.classList.add('hidden');
    const extractBtnEl = document.getElementById('extractBtn');
    if (extractBtnEl) extractBtnEl.disabled = true;
    const extractStatusEl = document.getElementById('extractStatus');
    if (extractStatusEl) extractStatusEl.textContent = 'Waiting for detection...';
    const compareStatusEl = document.getElementById('compareStatus');
    if (compareStatusEl) compareStatusEl.textContent = 'Step 1: Detect faces first';
    const compareBtnEl = document.getElementById('compareBtn');
    if (compareBtnEl) compareBtnEl.disabled = true;
    const comparisonResultEl = document.getElementById('comparisonResult');
    if (comparisonResultEl) comparisonResultEl.classList.add('hidden');
    visualizationData = {};
    showVisualizationPlaceholder();
    
    // Reset step states
    const step1El = document.getElementById('step1');
    if (step1El) step1El.classList.remove('step-complete');
    const step2El = document.getElementById('step2');
    if (step2El) step2El.classList.remove('step-complete');
    const step3El = document.getElementById('step3');
    if (step3El) step3El.classList.remove('step-complete');
    const step4El = document.getElementById('step4');
    if (step4El) step4El.classList.remove('step-complete');
    const webcamStepEl = document.getElementById('webcamStep');
    if (webcamStepEl) webcamStepEl.classList.remove('step-complete');
    
    // Reset button states
    const detectBtnEl = document.getElementById('detectBtn');
    if (detectBtnEl) detectBtnEl.classList.remove('btn-success');
    if (extractBtnEl) extractBtnEl.classList.remove('btn-success');
}

function markStepComplete(stepId, btnId) {
    const stepEl = document.getElementById(stepId);
    if (stepEl) stepEl.classList.add('step-complete');
    if (btnId) {
        const btnEl = document.getElementById(btnId);
        if (btnEl) {
            btnEl.classList.remove('btn-primary');
            btnEl.classList.add('btn-success');
        }
    }
}

function selectImage() {
    // Use imageInputAfter if it exists and is visible (after first upload)
    const imageInputAfter = document.getElementById('imageInputAfter');
    if (imageInputAfter && !imageInputAfter.parentElement.classList.contains('hidden')) {
        imageInputAfter.click();
    } else {
        document.getElementById('imageInput').click();
    }
}

function selectImageForStep5() {
    document.getElementById('step5ImageInput').click();
}

// Scroll to Step 1 then open file dialog
function scrollToStep1() {
    scrollToSection('step1');
    setTimeout(() => {
        selectImage();
    }, 300);
}

function addReference() {
    const refInputEl = document.getElementById('refInput');
    if (refInputEl) {
        refInputEl.click();
    } else {
        showToast('Reference input not found', 'error');
        logToTerminal('> Error: refInput element not found', 'error');
    }
}

function handleReferenceSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        saveReference(e.target.result, file.name);
    };
    reader.onerror = () => {
        logToTerminal('> Error reading reference file', 'error');
        showToast('Error reading file', 'error');
    };
    reader.readAsDataURL(file);
}

async function saveReference(imageData, name) {
    showLoading('Adding reference...');
    logToTerminal(`> Adding reference: ${name}`, 'command');

    try {
        logToTerminal('> Detecting face in reference image...', 'info');
        const response = await fetch(`${API_BASE}/add-reference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, name })
        });

        const data = await response.json();

        if (data.success) {
            logToTerminal(`> Reference "${name}" added successfully`, 'success');
            references.push(data.reference);
            updateReferenceList();

            // Enable compare button if we have an embedding
            if (currentQueryEmbedding !== null) {
                document.getElementById('compareBtn').disabled = false;
                document.getElementById('compareStatus').textContent = 'Step 4: Click "Compare" to find matches';
                await compareFaces();
            } else {
                document.getElementById('compareStatus').textContent = 'Step 3a: Extract features from your image first';
            }

            showToast(`Reference "${data.reference.name}" added`, 'success');
        } else {
            logToTerminal(`> Failed to add reference: ${data.error}`, 'error');
            showToast(data.error || 'Could not add reference', 'error');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

async function detectFaces() {
    if (!currentImage) return;

    showLoading('Detecting faces...');
    logToTerminal('> Loading image...', 'command');

    try {
        logToTerminal('> Sending image to AI model...', 'info');
        const response = await fetch(`${API_BASE}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImage })
        });

        const data = await response.json();
 
        if (data.success) {
            logToTerminal(`> Found ${data.count} face(s) in image`, 'success');
            
            // Update Step 5 status
            const step5Status = document.getElementById('step5Status');
            if (step5Status) {
                step5Status.textContent = `Found ${data.count} face(s)!`;
                step5Status.className = 'status status-success';
            }
            
            const extractBtnEl = document.getElementById('extractBtn');
            if (extractBtnEl) extractBtnEl.disabled = false;
            markStepComplete('step2', 'detectBtn');

            // Display preprocessing info
            if (data.preprocessing) {
                const prep = data.preprocessing;
                if (prep.was_enhanced) {
                    const msg = `Image enhanced: ${prep.method.toUpperCase()} (quality: ${(prep.enhanced_quality.overall * 100).toFixed(0)}%)`;
                    logToTerminal('> ' + msg, 'info');
                    if (step5Status) step5Status.textContent = `Found ${data.count} face(s) - ${prep.method} enhanced`;
                } else {
                    logToTerminal('> Image quality OK (no enhancement needed)', 'info');
                }
            }

            // Check for eyewear (sunglasses/glasses)
            try {
                const eyewearResponse = await fetch(`${API_BASE}/eyewear`);
                const eyewearData = await eyewearResponse.json();
                if (eyewearData.success && eyewearData.eyewear && eyewearData.eyewear.has_eyewear) {
                    const ew = eyewearData.eyewear;
                    const warningMsg = `⚠️ ${ew.type.toUpperCase()} detected (${Math.round(ew.confidence * 100)}% confidence) - may affect accuracy`;
                    logToTerminal('> ' + warningMsg, 'warning');
                    if (step5Status) {
                        step5Status.textContent = `Found ${data.count} face(s) - ${ew.type} detected!`;
                        step5Status.className = 'status status-warning';
                    }
                    showToast(warningMsg, 'warning');
                }
            } catch (ewErr) {
                console.log('[EYEWEAR] Check failed:', ewErr.message);
            }

            const gallery = document.getElementById('facesGallery');
            if (gallery) gallery.innerHTML = '';
            currentFaceThumbnails = data.faces;

            // Also show in Step 5
            const step5Gallery = document.getElementById('step5FacesGallery');
            if (step5Gallery) step5Gallery.innerHTML = '';

            data.faces.forEach((face, i) => {
                logToTerminal(`> Face ${i + 1}: bbox=[${face.bbox.join(', ')}]`, 'info');
                const div = document.createElement('div');
                div.className = 'gallery-item';
                div.innerHTML = `
                    <img src="data:image/png;base64,${face.thumbnail}" alt="Face ${i + 1}">
                    <span>Face ${i + 1}</span>
                `;
                if (gallery) gallery.appendChild(div);
                
                // Add to Step 5 gallery too
                const step5Div = document.createElement('div');
                step5Div.className = 'gallery-item';
                step5Div.innerHTML = `
                    <img src="data:image/png;base64,${face.thumbnail}" alt="Face ${i + 1}">
                    <span>Face ${i + 1}</span>
                `;
                if (step5Gallery) step5Gallery.appendChild(step5Div);
            });

            const facesContainerEl = document.getElementById('facesContainer');
            if (facesContainerEl) facesContainerEl.classList.add('visible');
            const step5FacesContainerEl = document.getElementById('step5FacesContainer');
            if (step5FacesContainerEl) {
                step5FacesContainerEl.classList.remove('hidden');
                step5FacesContainerEl.classList.add('visible');
            }

            Object.keys(data.visualizations).forEach(key => {
                visualizationData[key] = data.visualizations[key];
            });

            showVisualization('detection');
            showToast(`Found ${data.count} face(s)`, 'success');
        } else {
            logToTerminal('> No faces detected', 'error');
            const detectStatusEl = document.getElementById('detectStatus');
            if (detectStatusEl) {
                detectStatusEl.textContent = 'No faces detected';
                detectStatusEl.className = 'status status-warning';
            }
            const step5StatusEl = document.getElementById('step5Status');
            if (step5StatusEl) {
                step5StatusEl.textContent = 'No faces detected';
                step5StatusEl.className = 'status status-warning';
            }
            showToast(data.error || 'No faces detected', 'warning');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        const detectStatusEl = document.getElementById('detectStatus');
        if (detectStatusEl) {
            detectStatusEl.textContent = 'Error detecting faces';
            detectStatusEl.className = 'status status-error';
        }
        const step5StatusEl = document.getElementById('step5Status');
        if (step5StatusEl) {
            step5StatusEl.textContent = 'Error detecting faces';
            step5StatusEl.className = 'status status-error';
        }
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

async function extractFeatures() {
    if (currentFaceThumbnails.length === 0) return;

    showLoading('Extracting features...');
    logToTerminal('> Initializing feature extractor...', 'command');
    logToTerminal('> Processing face image...', 'info');

    try {
        logToTerminal('> Running FaceNet embedding extraction...', 'info');
        const response = await fetch(`${API_BASE}/extract`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ face_id: 0 })
        });

        const data = await response.json();

        if (data.success) {
            currentQueryEmbedding = data.embedding_mean;
            console.log('[EXTRACT] Received', Object.keys(data.visualizations).length, 'visualizations');
            console.log('[EXTRACT] Viz keys:', Object.keys(data.visualizations));
            
            Object.keys(data.visualizations).forEach(key => {
                visualizationData[key] = data.visualizations[key];
                console.log('[EXTRACT] Stored:', key, 'length:', data.visualizations[key]?.length);
            });
            
            if (data.visualization_data) {
                Object.keys(data.visualization_data).forEach(key => {
                    visualizationData[key + '_data'] = data.visualization_data[key];
                });
            }
            
            logToTerminal(`> Embedding vector: ${data.embedding_size} dimensions`, 'success');
            logToTerminal(`> Mean: ${data.embedding_mean.toFixed(6)}, Std: ${data.embedding_std.toFixed(6)}`, 'info');
            const extractStatusEl = document.getElementById('extractStatus');
            if (extractStatusEl) {
                extractStatusEl.textContent = `Features extracted (${data.embedding_size}-dim)`;
                extractStatusEl.className = 'status status-success';
            }
            const step5StatusEl = document.getElementById('step5Status');
            if (step5StatusEl) {
                step5StatusEl.textContent = `Signature created (${data.embedding_size}-dim)`;
                step5StatusEl.className = 'status status-success';
            }
            markStepComplete('step3', null);
            
            // Pre-fetch all visualization types in background
            const vizTypes = ['detection', 'extraction', 'preprocessing', 'landmarks', 'mesh3d', 'alignment', 'saliency', 'activations', 'features', 'multiscale', 'confidence', 'eyewear', 'iris', 'expression', 'embedding', 'similarity', 'robustness', 'biometric', 'asymmetry', 'texture', 'normalized'];
            logToTerminal('> Pre-caching all visualizations...', 'info');
            vizTypes.forEach(vizType => {
                if (!visualizationData[vizType]) {
                    fetch(`${API_BASE}/visualizations/${vizType}`)
                        .then(r => r.json())
                        .then(vizData => {
                            if (vizData.success && vizData.visualization) {
                                visualizationData[vizType] = vizData.visualization;
                                if (vizData.data) {
                                    visualizationData[vizType + '_data'] = vizData.data;
                                }
                                console.log('[CACHE] Cached:', vizType);
                            }
                        })
                        .catch(err => console.log('[CACHE] Failed:', vizType, err.message));
                }
            });
            
            // Enable compare button only if we have both embedding AND references
            const hasReferences = references && references.length > 0;
            const compareBtnEl = document.getElementById('compareBtn');
            if (compareBtnEl) {
                compareBtnEl.disabled = !hasReferences;
            }
            const compareStatusEl = document.getElementById('compareStatus');
            if (compareStatusEl) {
                if (hasReferences) {
                    compareStatusEl.textContent = 'Step 4: Click "Compare" to find matches';
                } else {
                    compareStatusEl.textContent = 'Step 3b: Add a reference image to compare';
                }
            }

            console.log('[EXTRACT] Cached visualizations:', Object.keys(visualizationData));
            
            showVisualization('embedding');
            showToast('Features extracted successfully', 'success');
            
            // Scroll to compare section
            setTimeout(() => {
                scrollToSection('step4');
            }, 500);
        } else {
            logToTerminal('> Feature extraction failed', 'error');
            const extractStatusEl = document.getElementById('extractStatus');
            if (extractStatusEl) {
                extractStatusEl.textContent = 'Extraction failed';
                extractStatusEl.className = 'status status-error';
            }
            const step5StatusEl = document.getElementById('step5Status');
            if (step5StatusEl) {
                step5StatusEl.textContent = 'Failed to create signature';
                step5StatusEl.className = 'status status-error';
            }
            showToast(data.error || 'Extraction failed', 'error');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        const extractStatusEl = document.getElementById('extractStatus');
        if (extractStatusEl) {
            extractStatusEl.textContent = 'Error extracting features';
            extractStatusEl.className = 'status status-error';
        }
        const step5StatusEl = document.getElementById('step5Status');
        if (step5StatusEl) {
            step5StatusEl.textContent = 'Error creating signature';
            step5StatusEl.className = 'status status-error';
        }
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
        checkLibraryCompareButton();
        checkFindMatchesButton();
    }
}

async function removeReference(index, btnOrEvent) {
    let btn = null;
    let event = null;
    
    if (btnOrEvent && btnOrEvent.target) {
        // It's an event object
        event = btnOrEvent;
        btn = btnOrEvent.target;
    } else if (btnOrEvent) {
        // It's a DOM element (this)
        btn = btnOrEvent;
    }
    
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    const ref = references[index];
    if (!ref) {
        logToTerminal(`> Error: Reference ${index} not found`, 'error');
        showToast('Reference not found', 'error');
        return;
    }
    
    const refName = ref.name || `Reference ${index + 1}`;
    logToTerminal(`> Removing: ${refName}`, 'info');
    
    if (btn) {
        btn.disabled = true;
        btn.style.opacity = '0.5';
    }
    
    try {
        const response = await fetch(`${API_BASE}/references/${index}`, { method: 'DELETE' });
        const data = await response.json();
        
        if (data.success) {
            references = references.filter((_, i) => i !== index);
            references.forEach((ref, i) => ref.id = i);
            updateReferenceList();
            
            // Close details modal if open
            const detailsPanel = document.getElementById('referenceDetails');
            if (detailsPanel && detailsPanel.classList.contains('active')) {
                detailsPanel.classList.remove('active');
            }
            
            logToTerminal(`> Removed: ${refName}`, 'success');
            showToast('Reference removed', 'success');
        } else {
            console.error('[REMOVE REF] API error:', data);
            throw new Error(data.error || 'Unknown error: ' + JSON.stringify(data));
        }
    } catch (err) {
        console.error('[REMOVE REF] Exception:', err);
        logToTerminal(`> Error removing ${refName}: ${err.message}`, 'error');
        showToast('Failed to remove reference: ' + err.message, 'error');
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = '1';
        }
    }
}

async function showReferenceVisualizations(refId) {
    const ref = references[refId];
    if (!ref) {
        logToTerminal(`> Error: Reference ${refId} not found`, 'error');
        showToast('Reference not found', 'error');
        return;
    }
    
    const refName = ref.name || `Reference ${refId + 1}`;
    logToTerminal(`> Loading visualizations for: ${refName}`, 'info');
    
    document.getElementById('vizContent').innerHTML = `
        <div class="viz-placeholder">
            <p>Loading ${refName}...</p>
        </div>
    `;
    
    showReferenceDetails(refId, ref);
    
    try {
        const response = await fetch(`${API_BASE}/visualizations/embedding/reference/${refId}`);
        const data = await response.json();
        
        if (data.success && data.visualization) {
            visualizationData[`ref_${refId}_embedding`] = data.visualization;
            showVisualization(`ref_${refId}_embedding`);
            logToTerminal(`> Showing embedding for: ${refName}`, 'success');
        } else {
            throw new Error(data.error || 'No embedding available');
        }
    } catch (err) {
        logToTerminal(`> Error loading ${refName}: ${err.message}`, 'error');
        document.getElementById('vizContent').innerHTML = `
            <div class="viz-placeholder">
                <p>No visualization available</p>
                <p class="text-error-inline">${err.message}</p>
            </div>
        `;
    }
}

function showReferenceDetailsOnly(refIndex, btnOrEvent) {
    let event = null;
    if (btnOrEvent && btnOrEvent.target) {
        event = btnOrEvent;
    }
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    const ref = references[refIndex];
    if (!ref) return;
    showReferenceDetails(refIndex, ref);
}

function showReferenceDetails(refId, ref) {
    const detailsPanel = document.getElementById('referenceDetails');
    const titleEl = document.getElementById('refDetailsTitle');
    const tabsEl = document.getElementById('refVizTabs');
    const contentEl = document.getElementById('refVizContent');
    const infoEl = document.getElementById('refInfoGrid');
    
    detailsPanel.classList.add('active');
    titleEl.textContent = ref.name || `Reference ${refId + 1}`;
    
    const tabs = [
        { id: 'info', label: 'Info' },
        { id: 'detection', label: 'Face' },
        { id: 'landmarks', label: 'Landmarks' },
        { id: 'iris', label: 'Iris' },
        { id: 'expression', label: 'Expression' },
        { id: 'embedding', label: 'Embedding' },
        { id: 'alignment', label: 'Pose' },
        { id: 'saliency', label: 'Attention' }
    ];
    
    // Generate radio buttons (hidden) + labels
    const tabCount = tabs.length;
    
    const radioHtml = tabs.map((t, i) => `
        <input type="radio" name="ref-viz" id="tab-${t.id}" 
               class="ref-viz-input" data-tab="${t.id}" 
               ${i === 0 ? 'checked' : ''} c-option="${i + 1}">
    `).join('');
    
    const labelHtml = tabs.map((t, i) => `
        <label class="ref-viz-tab ${i === 0 ? 'active' : ''}" 
               for="tab-${t.id}" data-tab="${t.id}">
            ${t.label}
        </label>
    `).join('');
    
    tabsEl.innerHTML = radioHtml + labelHtml;
    tabsEl.style.setProperty('--tab-count', tabCount);
    
    // Position indicator based on actual tab positions
    setTimeout(() => updateRefVizIndicator(tabsEl), 0);
    
    // Track previous for directional animation
    setTimeout(() => trackPrevious(tabsEl), 0);
    
    // Add change listeners to radios
    const radios = tabsEl.querySelectorAll('.ref-viz-input');
    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            const tabId = radio.dataset.tab;
            switchRefTab(tabId, refId);
        });
    });
    
    switchRefTab('info', refId);
}

function updateRefVizIndicator(tabsEl, activeTabId = null) {
    const tabs = tabsEl.querySelectorAll('.ref-viz-tab');
    const containerRect = tabsEl.getBoundingClientRect();
    const paddingLeft = 20;
    const gap = 8;
    
    // Calculate positions for each tab
    let tabPositions = [];
    tabs.forEach((tab, index) => {
        const tabRect = tab.getBoundingClientRect();
        const relativeLeft = tabRect.left - containerRect.left;
        tabPositions.push({
            id: tab.dataset.tab,
            left: relativeLeft,
            width: tabRect.width
        });
    });
    
    // If activeTabId provided, position the indicator
    if (activeTabId) {
        const active = tabPositions.find(t => t.id === activeTabId);
        if (active) {
            tabsEl.style.setProperty('--indicator-left', `${active.left}px`);
            tabsEl.style.setProperty('--indicator-width', `${active.width}px`);
        }
    }
    
    return tabPositions;
}

async function switchRefTab(tabId, refId) {
    const ref = references[refId];
    const tabs = document.querySelectorAll('.ref-viz-tab');
    const radios = document.querySelectorAll('.ref-viz-input');
    const tabsEl = document.getElementById('refVizTabs');
    
    tabs.forEach(t => t.classList.remove('active'));
    radios.forEach(r => r.checked = false);
    
    const activeTab = document.querySelector(`.ref-viz-tab[data-tab="${tabId}"]`);
    const activeRadio = document.querySelector(`.ref-viz-input[data-tab="${tabId}"]`);
    
    if (activeTab) activeTab.classList.add('active');
    if (activeRadio) activeRadio.checked = true;
    
    // Update indicator position based on actual tab position
    updateRefVizIndicator(tabsEl, tabId);
    
    const contentEl = document.getElementById('refVizContent');
    const infoEl = document.getElementById('refInfoGrid');
    
    if (tabId === 'info') {
        contentEl.innerHTML = `<img src="data:image/png;base64,${ref.thumbnail}" alt="Reference">`;
        
        const pose = ref.pose || {};
        const quality = ref.quality || {};
        
        infoEl.innerHTML = `
            <div class="ref-info-item">
                <div class="ref-info-label">Pose Yaw</div>
                <div class="ref-info-value">${pose.yaw?.toFixed(1) || '0'}°</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Pose Pitch</div>
                <div class="ref-info-value">${pose.pitch?.toFixed(1) || '0'}°</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Pose Roll</div>
                <div class="ref-info-value">${pose.roll?.toFixed(1) || '0'}°</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Pose Category</div>
                <div class="ref-info-value">${ref.pose_category || 'frontal'}</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Quality</div>
                <div class="ref-info-value">${quality.overall?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Brightness</div>
                <div class="ref-info-value">${quality.brightness?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Sharpness</div>
                <div class="ref-info-value">${quality.sharpness?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="ref-info-item">
                <div class="ref-info-label">Has Activations</div>
                <div class="ref-info-value">${ref.activations && Object.keys(ref.activations).length > 0 ? 'Yes' : 'No'}</div>
            </div>
        `;
    } else {
        // Fetch visualization from API
        contentEl.innerHTML = '<div class="viz-placeholder"><p>Loading...</p></div>';
        infoEl.innerHTML = '';
        
        try {
            const response = await fetch(`${API_BASE}/visualizations/${tabId}/reference/${refId}`);
            const data = await response.json();
            
            if (data.success && data.visualization) {
                contentEl.innerHTML = `<img src="data:image/png;base64,${data.visualization}" alt="${tabId}">`;
                if (data.data && Object.keys(data.data).length > 0) {
                    infoEl.innerHTML = '<div class="viz-data-table"><table><tbody>' + 
                        Object.entries(data.data).map(([key, value]) => 
                            `<tr><td class="label">${key}</td><td class="value">${value}</td></tr>`
                        ).join('') + 
                        '</tbody></table></div>';
                }
            } else {
                contentEl.innerHTML = `<img src="data:image/png;base64,${ref.thumbnail}" alt="Reference">`;
            }
        } catch (err) {
            contentEl.innerHTML = `<img src="data:image/png;base64,${ref.thumbnail}" alt="Reference">`;
        }
    }
}

function hideReferenceDetails() {
    document.getElementById('referenceDetails').classList.remove('active');
}

// Update currently uploaded section in Step 4
function updateCurrentlyUploaded() {
    const empty = document.querySelector('.currently-uploaded-empty');
    const preview = document.getElementById('currentlyUploadedPreview');
    const img = document.getElementById('currentlyUploadedImage');
    
    if (empty && preview && img) {
        if (currentImage) {
            empty.style.display = 'none';
            preview.classList.remove('hidden');
            preview.style.display = 'flex';
            img.src = currentImage;
        } else {
            empty.style.display = 'block';
            preview.classList.add('hidden');
            preview.style.display = 'none';
        }
    }
}

// Show currently uploaded image in reference details (reuse existing function)
function showUploadedAsReference() {
    if (!currentImage) {
        showToast('No image uploaded', 'warning');
        return;
    }
    
    // Scroll to reference details section in Step 4
    const refDetails = document.getElementById('referenceDetails');
    if (refDetails) {
        refDetails.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    // Reuse the reference details panel
    const details = document.getElementById('referenceDetails');
    const title = document.getElementById('refDetailsTitle');
    const vizTabs = document.getElementById('refVizTabs');
    
    title.textContent = 'Currently Uploaded - Details';
    details.classList.remove('hidden');
    details.classList.add('active');
    
    // Use same pattern as reference-details: radio buttons + labels for animation
    const tabs = [
        { id: 'detection', label: 'Detection' },
        { id: 'landmarks', label: 'Landmarks' },
        { id: 'mesh3d', label: '3D Mesh' },
        { id: 'iris', label: 'Iris' },
        { id: 'expression', label: 'Expression' },
        { id: 'alignment', label: 'Pose' },
        { id: 'embedding', label: 'Embedding' },
        { id: 'similarity', label: 'Similarity' },
        { id: 'eyewear', label: 'Eyewear' },
        { id: 'confidence', label: 'Quality' }
    ];
    
    const tabCount = tabs.length;
    
    const radioHtml = tabs.map((t, i) => `
        <input type="radio" name="uploaded-viz" id="uploaded-tab-${t.id}" 
               class="ref-viz-input" data-tab="${t.id}" 
               ${i === 0 ? 'checked' : ''}>
    `).join('');
    
    const labelHtml = tabs.map((t, i) => `
        <label class="ref-viz-tab ${i === 0 ? 'active' : ''}" 
               for="uploaded-tab-${t.id}" data-tab="${t.id}">
            ${t.label}
        </label>
    `).join('');
    
    vizTabs.innerHTML = radioHtml + labelHtml;
    vizTabs.style.setProperty('--tab-count', tabCount);
    
    // Position indicator based on actual tab positions
    setTimeout(() => updateRefVizIndicator(vizTabs), 0);
    setTimeout(() => trackPrevious(vizTabs), 0);
    
    // Add change listeners
    const radios = vizTabs.querySelectorAll('.ref-viz-input');
    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            const tabId = radio.dataset.tab;
            showUploadedVizContent(tabId);
            // Update indicator position
            setTimeout(() => updateRefVizIndicator(vizTabs), 10);
        });
    });
    
    // Show initial content
    showUploadedVizContent('detection');
    setTimeout(() => updateRefVizIndicator(vizTabs), 10);
}

function showUploadedVizContent(vizType) {
    const vizTabs = document.getElementById('refVizTabs');
    const vizContent = document.getElementById('refVizContent');
    const infoGrid = document.getElementById('refInfoGrid');
    
    // Update active tab styling
    vizTabs.querySelectorAll('.ref-viz-tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.tab === vizType) {
            tab.classList.add('active');
        }
    });
    
    // Update radio button
    const radio = vizTabs.querySelector(`#uploaded-tab-${vizType}`);
    if (radio) {
        radio.checked = true;
    }
    
    // Check if we have visualization data from detection
    if (vizType === 'detection' && visualizationData['detection']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['detection']}" alt="Detection" style="max-width:100%">`;
    } else if (vizType === 'embedding' && visualizationData['embedding']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['embedding']}" alt="Embedding" style="max-width:100%">`;
    } else if (vizType === 'similarity' && visualizationData['similarity']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['similarity']}" alt="Similarity" style="max-width:100%">`;
    } else if (vizType === 'landmarks' && visualizationData['landmarks']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['landmarks']}" alt="Landmarks" style="max-width:100%">`;
    } else if (vizType === 'mesh3d' && visualizationData['mesh3d']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['mesh3d']}" alt="3D Mesh" style="max-width:100%">`;
    } else if (vizType === 'alignment' && visualizationData['alignment']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['alignment']}" alt="Alignment" style="max-width:100%">`;
    } else if (vizType === 'eyewear' && visualizationData['eyewear']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['eyewear']}" alt="Eyewear" style="max-width:100%">`;
    } else if (vizType === 'confidence' && visualizationData['confidence']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['confidence']}" alt="Confidence" style="max-width:100%">`;
    } else if (vizType === 'iris' && visualizationData['iris']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['iris']}" alt="Iris" style="max-width:100%">`;
    } else if (vizType === 'expression' && visualizationData['expression']) {
        vizContent.innerHTML = `<img src="data:image/png;base64,${visualizationData['expression']}" alt="Expression" style="max-width:100%">`;
    } else {
        vizContent.innerHTML = '<div class="viz-placeholder"><p>Run detection first to see visualizations</p></div>';
    }
    
    infoGrid.innerHTML = '';
}

function updateReferenceList() {
    const container = document.getElementById('referenceList');
    container.innerHTML = '';
    
    if (!references || references.length === 0) {
        container.innerHTML = '<p class="empty-state">No references added yet</p>';
        return;
    }
    
    references.forEach((ref, i) => {
        if (!ref || !ref.thumbnail) {
            logToTerminal(`> Warning: Skipping corrupted reference at index ${i}`, 'warning');
            return;
        }
        
        const div = document.createElement('div');
        div.className = 'reference-item';
        div.onclick = () => showReferenceVisualizations(i);
        
        const name = ref.name || `Reference ${i + 1}`;
        
        div.innerHTML = `
            <div class="ref-remove-btn" data-index="${i}" onclick="removeReference(${i}, this)">×</div>
            <div class="ref-details-btn" data-index="${i}" onclick="showReferenceDetailsOnly(${i}, this)" title="View Details">i</div>
            <img src="data:image/png;base64,${ref.thumbnail}" alt="${name}">
            <span>${name}</span>
        `;
        container.appendChild(div);
    });
    
    // Also update sidebar refs
    updateSidebarRefs();
    
    // Also update Step 4 library list
    loadStep4Library();
}

// Load library persons into Step 4
async function loadStep4Library() {
    const container = document.getElementById('step4LibraryList');
    if (!container) return;
    
    try {
        const response = await fetch(`${API_BASE}/library`);
        const data = await response.json();
        
        if (!data.persons || data.persons.length === 0) {
            container.innerHTML = '<p class="empty-hint">No library persons. Add persons in Step 6.</p>';
            updateCompareButtons(false);
            return;
        }
        
        container.innerHTML = data.persons.map(person => {
            const thumbnail = person.first_image_thumbnail || '';
            
            return `
                <div class="library-ref-item" onclick="selectLibraryRefForCompare('${person.id}', '${person.name}')">
                    <img src="${thumbnail}" class="library-ref-thumb" alt="${person.name}">
                    <div class="library-ref-info">
                        <span class="library-ref-name">${person.name}</span>
                        <span class="library-ref-count">${person.image_count} image(s)</span>
                    </div>
                </div>
            `;
        }).join('');
        
        // Enable compare buttons since library has persons
        updateCompareButtons(true);
        
    } catch (err) {
        console.error('Failed to load Step 4 library:', err);
        container.innerHTML = '<p class="empty-hint">Error loading library</p>';
        updateCompareButtons(false);
    }
}

// Update compare buttons based on library status
function updateCompareButtons(hasLibraryPersons) {
    const compareBtn = document.getElementById('compareBtn');
    const compareLibraryBtn = document.getElementById('compareLibraryBtn');
    
    // Compare with Library button - enabled if library has persons
    if (compareLibraryBtn) {
        compareLibraryBtn.disabled = !hasLibraryPersons;
    }
    
    // Compare with Selected button - enabled if has references OR selected library refs
    if (compareBtn) {
        const hasRefs = references && references.length > 0;
        const hasSelected = selectedLibraryRefs && selectedLibraryRefs.length > 0;
        compareBtn.disabled = !hasRefs && !hasSelected;
        if (hasSelected) {
            compareBtn.textContent = `Compare with Selected (${selectedLibraryRefs.length})`;
        } else if (hasRefs) {
            compareBtn.textContent = 'Compare with Selected';
        }
    }
}

// Track selected library refs for comparison (multi-select via checkboxes)
let selectedLibraryRefs = [];
let selectedLibraryRefNames = [];

async function loadStep4Library() {
    const container = document.getElementById('step4LibraryList');
    if (!container) return;
    
    try {
        const response = await fetch(`${API_BASE}/library`);
        const data = await response.json();
        
        if (!data.persons || data.persons.length === 0) {
            container.innerHTML = '<p class="empty-hint">No library persons. Add persons in Step 6.</p>';
            updateCompareButtons(false);
            return;
        }
        
        container.innerHTML = data.persons.map((person, index) => {
            const thumbnail = person.first_image_thumbnail || '';
            const checkboxId = `lib-ref-${person.id}`;
            
            return `
                <div class="library-ref-item" onclick="event.stopPropagation(); toggleLibraryRefForCompare('${person.id}', '${person.name}')">
                    <input type="checkbox" name="libraryRef" id="${checkboxId}" value="${person.id}" class="library-ref-checkbox" onchange="toggleLibraryRefForCompare('${person.id}', '${person.name}')">
                    <label for="${checkboxId}" class="library-ref-label">
                        <img src="${thumbnail}" class="library-ref-thumb" alt="${person.name}">
                        <div class="library-ref-info">
                            <span class="library-ref-name">${person.name}</span>
                            <span class="library-ref-count">${person.image_count} image(s)</span>
                        </div>
                    </label>
                </div>
            `;
        }).join('');
        
        // Enable compare buttons since library has persons
        updateCompareButtons(true);
        
    } catch (err) {
        console.error('Failed to load Step 4 library:', err);
        container.innerHTML = '<p class="empty-hint">Error loading library</p>';
        updateCompareButtons(false);
    }
}

function toggleLibraryRefForCompare(personId, personName) {
    // Toggle selection
    const index = selectedLibraryRefs.indexOf(personId);
    if (index > -1) {
        selectedLibraryRefs.splice(index, 1);
        selectedLibraryRefNames.splice(index, 1);
    } else {
        selectedLibraryRefs.push(personId);
        selectedLibraryRefNames.push(personName);
    }
    
    // Update checkbox state
    const checkboxId = `lib-ref-${personId}`;
    const checkbox = document.getElementById(checkboxId);
    if (checkbox) checkbox.checked = index === -1;
    
    // Update UI to show selection highlight
    const items = document.querySelectorAll('.library-ref-item');
    items.forEach(item => {
        const checkbox = item.querySelector('input[type="checkbox"]');
        if (checkbox && checkbox.checked) {
            item.classList.add('selected');
        } else if (checkbox) {
            item.classList.remove('selected');
        }
    });
    
    // Enable/disable compare button
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        const hasSelected = selectedLibraryRefs.length > 0;
        const hasTempRefs = references && references.length > 0;
        compareBtn.disabled = !hasSelected && !hasTempRefs;
        if (selectedLibraryRefs.length > 0) {
            compareBtn.textContent = `Compare with Selected (${selectedLibraryRefs.length})`;
        } else {
            compareBtn.textContent = 'Compare with Selected';
        }
    }
    
    logToTerminal(`> Selected ${selectedLibraryRefs.length} library ref(s): ${selectedLibraryRefNames.join(', ')}`, 'info');
}

function updateSidebarRefs() {
    const grid = document.getElementById('sidebarRefsGrid');
    const countEl = document.getElementById('refCount');
    
    if (!grid) return;
    
    if (!references || references.length === 0) {
        grid.innerHTML = '<small style="opacity:0.5">No references yet</small>';
        if (countEl) countEl.textContent = '0';
        return;
    }
    
    if (countEl) countEl.textContent = references.length;
    
    // Show up to 8 most recent references
    const recentRefs = references.slice(-8);
    
    grid.innerHTML = recentRefs.map((ref, i) => {
        if (!ref || !ref.thumbnail) return '';
        const name = ref.name || `Ref ${i + 1}`;
        return `
            <div class="ref-thumb" onclick="jumpToStep(2)" title="${name}">
                <img src="data:image/png;base64,${ref.thumbnail}" alt="${name}">
            </div>
        `;
    }).join('');
}

function selectReference(index) {
    selectedReferenceId = index;
}

// Clear comparison results
function clearComparisonResults() {
    const resultEl = document.getElementById('comparisonResult');
    const statusEl = document.getElementById('matchStatus');
    const queryEl = document.getElementById('queryImage');
    const refEl = document.getElementById('refImage');
    const refLabelEl = document.getElementById('refLabel');
    
    // Reset all score displays
    const scoreIds = ['arcfaceScore', 'facenetScore', 'normScore', 'multiPoseScore', 
                      'lbpScore', 'asymScore', 'activationScore', 'irisScore', 
                      'expressionScore', 'matchScore'];
    scoreIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '--%';
    });
    
    // Hide result container
    if (resultEl) {
        resultEl.classList.remove('visible', 'active');
    }
    
    // Reset images (clear src but keep visible for flex layout)
    if (queryEl) {
        queryEl.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    }
    if (refEl) {
        refEl.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    }
    if (refLabelEl) refLabelEl.textContent = '';
    if (statusEl) {
        statusEl.textContent = '--';
        statusEl.className = 'comparison-status';
    }
    
    // Collapse scores dropdown
    const toggle = document.querySelector('.scores-toggle');
    const scores = document.querySelector('.comparison-scores');
    if (toggle && scores) {
        toggle.setAttribute('data-expanded', 'false');
        scores.setAttribute('data-visible', 'false');
        toggle.querySelector('.toggle-icon').textContent = '▶';
    }
}

// Compare Faces
async function compareFaces() {
    // Clear previous results
    clearComparisonResults();
    
    console.log('[COMPARE] currentQueryEmbedding:', currentQueryEmbedding ? 'yes' : 'no');
    console.log('[COMPARE] selectedLibraryRefs:', selectedLibraryRefs);
    console.log('[COMPARE] references.length:', references?.length || 0);
    logToTerminal(`> Compare: embedding=${currentQueryEmbedding ? 'yes' : 'no'}, libraryRefs=${selectedLibraryRefs?.length || 0}, tempRefs=${references?.length || 0}`, 'info');

    if (currentQueryEmbedding === null) {
        logToTerminal('> Error: No embedding extracted. Click "Create Signature" first.', 'error');
        showToast('Extract features first!', 'error');
        return;
    }
    
    // Check if we have library refs selected or temporary refs
    const hasLibraryRefs = selectedLibraryRefs && selectedLibraryRefs.length > 0;
    const hasTempRefs = references && references.length > 0;
    
    if (!hasLibraryRefs && !hasTempRefs) {
        logToTerminal('> Error: No reference selected. Select a library person or upload a reference.', 'error');
        showToast('Select a reference to compare', 'warning');
        return;
    }

    showLoading('Comparing...');
    logToTerminal('> Initializing similarity comparison...', 'command');
    
    let data;
    
    try {
        // If library refs selected, use the library compare endpoint
        if (hasLibraryRefs) {
            logToTerminal(`> Comparing against ${selectedLibraryRefs.length} library person(s): ${selectedLibraryRefNames.join(', ')}...`, 'info');
            console.log('[COMPARE] Using library ref IDs:', selectedLibraryRefs);
            
            // Compare with each selected library person and get best match
            let bestOverall = null;
            let bestScore = -1;
            
            for (let i = 0; i < selectedLibraryRefs.length; i++) {
                const personId = selectedLibraryRefs[i];
                const personName = selectedLibraryRefNames[i];
                
                const response = await fetch(`${API_BASE}/compare/library/${encodeURIComponent(personId)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const personData = await response.json();
                
                console.log(`[COMPARE] ${personName} response:`, personData);
                
                if (personData.success && personData.best_match) {
                    const score = personData.best_match.final_score || 0;
                    if (score > bestScore) {
                        bestScore = score;
                        bestOverall = personData.best_match;
                    }
                }
            }
            
            if (bestOverall) {
                data = { success: true, best_match: bestOverall };
            } else {
                data = { success: false, error: 'No matches found' };
            }
        } else {
            // Use temporary references
            logToTerminal(`> Comparing against ${references.length} temporary reference(s)...`, 'info');
            const response = await fetch(`${API_BASE}/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            data = await response.json();
        }

        if (data.success && data.best_match) {
            const best = data.best_match;

            logToTerminal(`> Best match: ${best.name}`, 'success');
            logToTerminal(`> Final score: ${(best.final_score * 100).toFixed(1)}%`, 'success');
            logToTerminal(`> Match status: ${best.match_label}`, 'success');
            if (best.arcface_similarity !== null) {
                logToTerminal(`> ArcFace: ${(best.arcface_similarity * 100).toFixed(1)}%`, 'info');
            }
            if (best.facenet_similarity !== null) {
                logToTerminal(`> FaceNet: ${(best.facenet_similarity * 100).toFixed(1)}%`, 'info');
            }

            // Update comparison result - with null checks
            const queryImageEl = document.getElementById('queryImage');
            const refImageEl = document.getElementById('refImage');
            const refLabelEl = document.getElementById('refLabel');
            const statusEl = document.getElementById('matchStatus');
            const comparisonResultEl = document.getElementById('comparisonResult');
            
            // Use query_thumbnail from compare response, fallback to currentFaceThumbnails
            const queryThumb = data.query_thumbnail || (currentFaceThumbnails[0] && currentFaceThumbnails[0].thumbnail);
            console.log('[COMPARE] queryThumb:', queryThumb ? 'has value' : 'empty', '| currentFaceThumbnails:', currentFaceThumbnails?.length);
            if (queryImageEl) {
                if (queryThumb) {
                    queryImageEl.src = `data:image/png;base64,${queryThumb}`;
                    queryImageEl.style.display = 'inline-block';
                } else if (currentImage) {
                    queryImageEl.src = currentImage;
                    queryImageEl.style.display = 'inline-block';
                }
            } else {
                console.log('[COMPARE] queryImageEl not found!');
            }
            if (refImageEl && best.thumbnail) {
                refImageEl.src = `data:image/png;base64,${best.thumbnail}`;
                refImageEl.style.display = 'inline-block';
            }
            if (refLabelEl) refLabelEl.textContent = best.name;
            
            // Display match status
            if (statusEl) {
                statusEl.textContent = best.match_label;
                statusEl.className = `comparison-status ${best.status}`;
            }
            
            // Display ArcFace score
            const arcfaceEl = document.getElementById('arcfaceScore');
            console.log('[COMPARE] best.arcface_similarity:', best.arcface_similarity, 'type:', typeof best.arcface_similarity);
            if (arcfaceEl && best.arcface_similarity != null) {
                const score = parseFloat(best.arcface_similarity);
                if (!isNaN(score)) {
                    arcfaceEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    arcfaceEl.textContent = 'N/A';
                }
            }
            
            // Display FaceNet score
            const facenetEl = document.getElementById('facenetScore');
            if (facenetEl && best.facenet_similarity != null) {
                const score = parseFloat(best.facenet_similarity);
                if (!isNaN(score)) {
                    facenetEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    facenetEl.textContent = 'N/A';
                }
            }
            
            // Display Activation similarity score
            const activationEl = document.getElementById('activationScore');
            if (activationEl && best.activation_similarity != null) {
                const score = parseFloat(best.activation_similarity);
                if (!isNaN(score)) {
                    activationEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    activationEl.textContent = 'N/A';
                }
            }
            
            // Display Iris similarity score
            const irisEl = document.getElementById('irisScore');
            if (irisEl && best.iris_similarity != null) {
                const score = parseFloat(best.iris_similarity);
                if (!isNaN(score)) {
                    irisEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    irisEl.textContent = 'N/A';
                }
            }
            
            // Display Expression similarity score
            const exprEl = document.getElementById('expressionScore');
            if (exprEl && best.expression_similarity != null) {
                const score = parseFloat(best.expression_similarity);
                if (!isNaN(score)) {
                    exprEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    exprEl.textContent = 'N/A';
                }
            }
            
            // Display 3D Normalized score
            const normEl = document.getElementById('normScore');
            if (normEl && best.normalized_similarity != null) {
                const score = parseFloat(best.normalized_similarity);
                if (!isNaN(score)) {
                    normEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    normEl.textContent = 'N/A';
                }
            }
            
            // Display Multi-Pose score
            const multiPoseEl = document.getElementById('multiPoseScore');
            if (multiPoseEl && best.multi_pose_score != null) {
                const score = parseFloat(best.multi_pose_score);
                if (!isNaN(score)) {
                    multiPoseEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    multiPoseEl.textContent = 'N/A';
                }
            }
            
            // Display Texture (LBP) score
            const lbpEl = document.getElementById('lbpScore');
            if (lbpEl && best.lbp_similarity != null) {
                const score = parseFloat(best.lbp_similarity);
                if (!isNaN(score)) {
                    lbpEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    lbpEl.textContent = 'N/A';
                }
            }
            
            // Display Uniqueness (Asymmetry) score
            const asymEl = document.getElementById('asymScore');
            if (asymEl && best.asymmetry_similarity != null) {
                const score = parseFloat(best.asymmetry_similarity);
                if (!isNaN(score)) {
                    asymEl.textContent = `${Math.round(score * 100)}%`;
                } else {
                    asymEl.textContent = 'N/A';
                }
            }
            
            // Display reasons
            const reasonsEl = document.getElementById('matchReasons');
            if (reasonsEl) {
                if (best.reasons && best.reasons.length > 0) {
                    reasonsEl.innerHTML = `
                        <div class="match-reasons-toggle" onclick="const isExpanded = this.getAttribute('data-expanded') === 'true'; this.setAttribute('data-expanded', (!isExpanded).toString()); this.nextElementSibling.setAttribute('data-visible', (!isExpanded).toString())">
                            <span class="toggle-icon">▶</span>
                            <span>Show details (${best.reasons.length})</span>
                        </div>
                        <div class="match-reasons-content">
                            <ul>${best.reasons.map(r => `<li>${r}</li>`).join('')}</ul>
                        </div>
                    `;
                } else {
                        reasonsEl.innerHTML = '';
                }
            }

            // Show comparison result container
            if (comparisonResultEl) {
                comparisonResultEl.classList.remove('hidden');
                comparisonResultEl.classList.add('visible');
                comparisonResultEl.classList.add('active');
                comparisonResultEl.style.display = 'flex';
            }
            
            // Auto-expand scores dropdown
            expandScoresDropdown();
            
            if (compareStatusEl) {
                compareStatusEl.textContent = `Best match: ${best.name} (${Math.round(best.final_score * 100)}%)`;
                compareStatusEl.className = 'status status-success';
            }

            // Store similarity visualization
            visualizationData['similarity'] = data.similarity_viz;
            visualizationData['similarity_data'] = data.similarity_data;
            
            showVisualization('similarity');

            // Mark step 4 as complete
            markStepComplete('step4', 'compareBtn');

            showToast(`${best.match_label}: ${best.name} (${Math.round(best.final_score * 100)}%)`, 'success');
        } else {
            const errorMsg = data.error || 'No match found';
            logToTerminal(`> ${errorMsg}`, 'warning');
            const compareStatusEl = document.getElementById('compareStatus');
            if (compareStatusEl) {
                compareStatusEl.textContent = errorMsg;
                compareStatusEl.className = 'status status-warning';
            }
            showToast(errorMsg, 'warning');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        const compareStatusEl = document.getElementById('compareStatus');
        if (compareStatusEl) {
            compareStatusEl.textContent = 'Error comparing';
            compareStatusEl.className = 'status status-error';
        }
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// Visualization
async function showVisualization(vizType) {
    const content = document.getElementById('vizContent');
    if (!content) {
        console.error('[VIZ] vizContent element not found');
        return;
    }
    
    console.log('[VIZ] Requested:', vizType, 'currentFaceThumbnails:', currentFaceThumbnails?.length, 'currentQueryEmbedding:', currentQueryEmbedding ? 'yes' : 'no');
    console.log('[VIZ] Available in cache:', Object.keys(visualizationData));
    
    // Check if we have the required data
    if (!currentFaceThumbnails || currentFaceThumbnails.length === 0) {
        content.innerHTML = `
            <div class="viz-placeholder">
                <p class="text-error-inline">No face detected</p>
                <p>1. Upload an image</p>
                <p>2. Click "Find Faces"</p>
                <p>3. Click "Create Signature"</p>
                <p>4. Then click visualization tabs</p>
            </div>
        `;
        logToTerminal(`> No face detected - run detection first`, 'warning');
        return;
    }
    
    if (!currentQueryEmbedding) {
        content.innerHTML = `
            <div class="viz-placeholder">
                <p class="text-error-inline">No embedding extracted</p>
                <p>Click "Create Signature" first to extract features</p>
            </div>
        `;
        logToTerminal(`> No embedding - click "Create Signature" first`, 'warning');
        return;
    }
    
    console.log('[VIZ] In cache:', visualizationData[vizType] ? 'YES' : 'NO');
    logToTerminal(`> Loading visualization: ${vizType}`, 'info');

    // If data not available locally, fetch from API
    if (!visualizationData[vizType]) {
        try {
            console.log('[VIZ] Fetching from API:', `${API_BASE}/visualizations/${vizType}`);
            logToTerminal(`> Fetching ${vizType} from API...`, 'info');
            const response = await fetch(`${API_BASE}/visualizations/${vizType}`);
            const data = await response.json();
            console.log('[VIZ] API response:', data);

            if (data.success && data.visualization) {
                visualizationData[vizType] = data.visualization;
                if (data.data && Object.keys(data.data).length > 0) {
                    visualizationData[vizType + '_data'] = data.data;
                }
                logToTerminal(`> Received ${data.visualization.length} chars for ${vizType}`, 'success');
            } 
            
            if (!data.visualization && (!data.data || Object.keys(data.data).length === 0)) {
                console.log('[VIZ] API returned no/invalid data:', data);
                content.innerHTML = `
                    <div class="viz-placeholder">
                        <p class="text-error-inline">No ${vizType} data available</p>
                        <p>Try running the full workflow:</p>
                        <p>1. Upload image → 2. Find Faces → 3. Create Signature</p>
                    </div>
                `;
                return;
            }
        } catch (err) {
            logToTerminal(`> Failed to fetch ${vizType}: ${err.message}`, 'error');
            console.log('[VIZ] Fetch error:', err);
            content.innerHTML = `
                <div class="viz-placeholder">
                    <p class="text-error-inline">Error loading ${vizType}</p>
                    <p>${err.message}</p>
                </div>
            `;
            return;
        }
    }

    if (visualizationData[vizType]) {
        const length = visualizationData[vizType].length;
        console.log('[VIZ] Displaying:', vizType, 'length:', length);
        logToTerminal(`> Displaying ${vizType} (${length} chars)`, 'success');

        let html = `<img src="data:image/png;base64,${visualizationData[vizType]}" alt="${vizType}">`;

        // Add data table if available
        const dataKey = vizType + '_data';
        if (visualizationData[dataKey]) {
            html += formatDataAsTable(visualizationData[dataKey]);
        }

        content.innerHTML = html;
    } else {
        // Check if we have data without image
        const dataKey = vizType + '_data';
        if (visualizationData[dataKey]) {
            logToTerminal(`> Displaying data only for ${vizType}`, 'success');
            content.innerHTML = formatDataAsTable(visualizationData[dataKey]);
        } else {
            logToTerminal(`> No data for ${vizType}`, 'warning');
            console.log('[VIZ] No data found for:', vizType);
            content.innerHTML = `
                <div class="viz-placeholder">
                    <p>No ${vizType} data available</p>
                    <p>Run: Upload → Find Faces → Create Signature</p>
                </div>
            `;
        }
    }
}

function formatDataAsTable(data) {
    if (!data || Object.keys(data).length === 0) return '';
    
    let html = '<div class="viz-data-table"><table>';
    
    if (typeof data === 'object' && !Array.isArray(data)) {
        // Dictionary object
        html += '<tbody>';
        for (const [key, value] of Object.entries(data)) {
            const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
            html += `<tr><td class="label">${formatKey(key)}</td><td class="value">${displayValue}</td></tr>`;
        }
        html += '</tbody>';
    } else if (Array.isArray(data)) {
        // Array
        html += '<tbody>';
        data.forEach((item, i) => {
            html += `<tr><td class="label">${i}</td><td class="value">${typeof item === 'number' ? item.toFixed(4) : item}</td></tr>`;
        });
        html += '</tbody>';
    }
    
    html += '</table></div>';
    return html;
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim();
}

function showVisualizationPlaceholder() {
    document.getElementById('vizContent').innerHTML = `
        <div class="viz-placeholder">
            <p>Run analysis to see visualizations</p>
        </div>
    `;
}

// Loading
function showLoading(text) {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// Expand scores dropdown
function expandScoresDropdown() {
    const toggle = document.querySelector('.scores-toggle');
    const scores = document.querySelector('.comparison-scores');
    if (toggle && scores) {
        toggle.setAttribute('data-expanded', 'true');
        scores.setAttribute('data-visible', 'true');
        toggle.querySelector('.toggle-icon').textContent = '▼';
    }
}

// Toast
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Webcam Functions
let webcamStream = null;
let faceMesh = null;
let meshCamera = null;
let meshOverlayActive = false;

async function startWebcam() {
    const video = document.getElementById('webcamVideo');
    const container = document.getElementById('webcamContainer');
    const status = document.getElementById('webcamStatus');
    const startBtn = document.getElementById('startWebcamBtn');
    const captureBtn = document.getElementById('captureWebcamBtn');
    const stopBtn = document.getElementById('stopWebcamBtn');

    try {
        logToTerminal('> Starting webcam...', 'info');
        status.textContent = 'Requesting camera access...';
        
        container.classList.remove('hidden');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        webcamStream = stream;
        video.srcObject = stream;
        container.classList.add('visible');
        document.getElementById('webcamStep').classList.add('visible');
        
        // Set webcam active flag
        currentWebcamActive = true;
        
        startBtn.disabled = true;
        captureBtn.disabled = false;
        stopBtn.disabled = false;
        
        const toggleMeshBtn = document.getElementById('toggleMeshBtn');
        if (toggleMeshBtn) {
            toggleMeshBtn.disabled = false;
        }
        
        const profileBtn = document.getElementById('profileCaptureBtn');
        if (profileBtn) {
            profileBtn.disabled = false;
        }
        
        const autoCaptureBtn = document.getElementById('autoCaptureBtn');
        if (autoCaptureBtn) {
            autoCaptureBtn.disabled = false;
        }
        
        status.textContent = 'Webcam active - Click "Capture" to take a photo';
        status.className = 'status status-success';
        logToTerminal('> Webcam started successfully', 'success');
        showToast('Webcam started', 'success');
        
    } catch (err) {
        logToTerminal(`> Webcam error: ${err.message}`, 'error');
        status.textContent = 'Error: ' + err.message;
        status.className = 'status status-error';
        showToast('Failed to start webcam: ' + err.message, 'error');
    }
}

// Profile capture - capture multiple angles and save as one person
let profileCaptures = [];

function startProfileCapture() {
    profileCaptures = [];
    const status = document.getElementById('webcamStatus');
    const btn = document.getElementById('profileCaptureBtn');
    
    if (!currentWebcamActive) {
        showToast('Start webcam first', 'warning');
        return;
    }
    
    logToTerminal('> Starting profile capture mode...', 'info');
    status.textContent = 'Profile Mode: Capturing 5 angles...';
    btn.disabled = true;
    btn.textContent = 'Capturing...';
    
    // Capture 5 images with 2 second intervals
    const captureInterval = setInterval(() => {
        if (profileCaptures.length >= 5) {
            clearInterval(captureInterval);
            finishProfileCapture();
            return;
        }
        
        captureWebcamForProfile();
    }, 2000);
    
    // Capture first one immediately
    captureWebcamForProfile();
}

function captureWebcamForProfile() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.createElement('canvas');
    
    if (!video.srcObject) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    profileCaptures.push(dataUrl);
    
    const status = document.getElementById('webcamStatus');
    status.textContent = `Profile: Captured ${profileCaptures.length}/5 angles`;
    logToTerminal(`> Captured angle ${profileCaptures.length}/5`, 'info');
}

function finishProfileCapture() {
    const btn = document.getElementById('profileCaptureBtn');
    const status = document.getElementById('webcamStatus');
    
    btn.disabled = false;
    btn.textContent = 'Capture Profile';
    
    if (profileCaptures.length === 0) {
        showToast('No captures taken', 'warning');
        return;
    }
    
    logToTerminal(`> Profile capture complete: ${profileCaptures.length} images`, 'success');
    status.textContent = `Profile captured: ${profileCaptures.length} angles - Opening library...`;
    
    // Open library modal to add this profile
    openProfileCaptureModal();
}

function openProfileCaptureModal() {
    // Store captured images globally for the modal to use
    window.profileCaptureImages = [...profileCaptures];
    
    // Show modal with name input
    const name = prompt('Enter person name for this profile:', 'Person');
    if (!name) {
        showToast('Name required', 'warning');
        return;
    }
    
    // Save to library
    saveProfileToLibrary(name);
}

async function saveProfileToLibrary(name) {
    showLoading('Saving profile to library...');
    
    try {
        // Save the profile images to library one by one
        for (let i = 0; i < profileCaptures.length; i++) {
            const imgData = profileCaptures[i];
            
            const response = await fetch(`${API_BASE}/library/person`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    image: imgData.split(',')[1]
                })
            });
            
            const data = await response.json();
            if (!data.success && i === 0) {
                throw new Error(data.error || 'Failed to save');
            }
        }
        
        showToast(`Profile "${name}" saved with ${profileCaptures.length} images`, 'success');
        logToTerminal(`> Profile saved: ${name} (${profileCaptures.length} angles)`, 'success');
        loadLibrary();
        
    } catch (err) {
        logToTerminal(`> Error saving profile: ${err.message}`, 'error');
        showToast('Error saving profile: ' + err.message, 'error');
    }
    
    profileCaptures = [];
    hideLoading();
}

function captureWebcam() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const status = document.getElementById('webcamStatus');
    
    if (!video.srcObject) {
        logToTerminal('> No webcam stream active', 'error');
        showToast('Start webcam first', 'warning');
        return;
    }
    
    logToTerminal('> Capturing frame from webcam...', 'info');
    status.textContent = 'Capturing...';
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    
    const byteString = atob(dataUrl.split(',')[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([ab], { type: 'image/jpeg' });
    const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
    
    const mockEvent = {
        target: {
            files: [file]
        }
    };
    
    logToTerminal('> Frame captured, processing...', 'info');
    handleImageSelect(mockEvent);
    
    // Show capture actions and enable library buttons
    showCaptureActions();
    checkLibraryCompareButton();
    checkFindMatchesButton();
    
    status.textContent = 'Photo captured! Continue with Step 2';
    showToast('Photo captured from webcam', 'success');
}

function stopWebcam() {
    const video = document.getElementById('webcamVideo');
    const container = document.getElementById('webcamContainer');
    const status = document.getElementById('webcamStatus');
    const startBtn = document.getElementById('startWebcamBtn');
    const captureBtn = document.getElementById('captureWebcamBtn');
    const stopBtn = document.getElementById('stopWebcamBtn');
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    video.srcObject = null;
    
    container.classList.remove('visible');
    container.classList.add('hidden');
    document.getElementById('webcamStep').classList.remove('visible');
    
    startBtn.disabled = false;
    captureBtn.disabled = true;
    stopBtn.disabled = true;
    
    const toggleMeshBtn = document.getElementById('toggleMeshBtn');
    if (toggleMeshBtn) {
        toggleMeshBtn.disabled = true;
        toggleMeshBtn.textContent = 'Show Mesh';
    }
    
    const profileBtn = document.getElementById('profileCaptureBtn');
    if (profileBtn) {
        profileBtn.disabled = true;
    }
    
    const autoCaptureBtn = document.getElementById('autoCaptureBtn');
    if (autoCaptureBtn) {
        autoCaptureBtn.disabled = true;
    }
    
    status.textContent = 'Webcam stopped';
    status.className = 'status';
    logToTerminal('> Webcam stopped', 'info');
    
    if (meshOverlayActive) {
        toggleMeshOverlay();
    }
}

// Auto-Capture Functions
let autoCapturedFrames = [];
let autoCaptureInProgress = false;

async function startAutoCapture() {
    if (autoCaptureInProgress) {
        showToast('Auto-capture already in progress', 'warning');
        return;
    }
    
    if (!webcamStream) {
        showToast('Start webcam first', 'warning');
        return;
    }
    
    autoCaptureInProgress = true;
    autoCapturedFrames = [];
    
    const status = document.getElementById('webcamStatus');
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const container = document.getElementById('webcamContainer');
    
    status.textContent = 'Auto-capturing 5 frames...';
    logToTerminal('> Starting auto-capture (5 frames)...', 'command');
    
    for (let i = 0; i < 5; i++) {
        // Show green flash
        showCaptureFlash(container);
        
        // Capture frame
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        autoCapturedFrames.push(dataUrl);
        
        status.textContent = `Captured ${i + 1}/5 frames...`;
        logToTerminal(`> Captured frame ${i + 1}/5`, 'info');
        
        // Wait 500ms before next capture
        if (i < 4) {
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    
    autoCaptureInProgress = false;
    status.textContent = `Captured ${autoCapturedFrames.length} frames`;
    logToTerminal(`> Auto-capture complete: ${autoCapturedFrames.length} frames`, 'success');
    showToast(`Captured ${autoCapturedFrames.length} frames`, 'success');
    
    // Show gallery modal to review and name
    showCaptureGalleryModal();
}

function showCaptureFlash(container) {
    // Add flash class
    container.classList.add('capture-flash');
    
    // Remove after animation
    setTimeout(() => {
        container.classList.remove('capture-flash');
    }, 300);
}

function showCaptureGalleryModal() {
    // Create modal if it doesn't exist
    let modal = document.getElementById('captureGalleryModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'captureGalleryModal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content capture-gallery-modal">
                <div class="modal-header">
                    <h3>Review Captures</h3>
                    <button class="btn-close-liquid" onclick="closeCaptureGalleryModal()">×</button>
                </div>
                <div class="modal-body">
                    <div class="capture-gallery" id="captureGalleryGrid"></div>
                    <div class="form-group">
                        <label>Person Name *</label>
                        <input type="text" id="captureGalleryName" placeholder="Enter name">
                    </div>
                    <div class="form-group">
                        <label>Notes (optional)</label>
                        <textarea id="captureGalleryNotes" placeholder="Optional notes"></textarea>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn" onclick="closeCaptureGalleryModal()">Cancel</button>
                    <button class="btn" onclick="retakeCaptureGallery()">Retake</button>
                    <button class="btn btn-primary" onclick="saveCaptureGalleryToLibrary()">Save to Library</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
    
    // Populate gallery
    const grid = document.getElementById('captureGalleryGrid');
    grid.innerHTML = autoCapturedFrames.map((frame, i) => `
        <div class="capture-gallery-item">
            <img src="${frame}" alt="Capture ${i + 1}">
            <span class="capture-number">${i + 1}</span>
        </div>
    `).join('');
    
    // Clear inputs
    document.getElementById('captureGalleryName').value = '';
    document.getElementById('captureGalleryNotes').value = '';
    
    // Show modal
    modal.classList.remove('hidden');
    modal.classList.add('active');
}

function closeCaptureGalleryModal() {
    const modal = document.getElementById('captureGalleryModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('active');
    }
}

function retakeCaptureGallery() {
    closeCaptureGalleryModal();
    autoCapturedFrames = [];
    startAutoCapture();
}

async function saveCaptureGalleryToLibrary() {
    const nameInput = document.getElementById('captureGalleryName');
    const notesInput = document.getElementById('captureGalleryNotes');
    const name = nameInput.value.trim();
    const notes = notesInput.value.trim();
    
    if (!name) {
        showToast('Please enter a name', 'warning');
        nameInput.focus();
        return;
    }
    
    showLoading(`Saving ${autoCapturedFrames.length} frames to library...`);
    
    try {
        let successCount = 0;
        
        for (let i = 0; i < autoCapturedFrames.length; i++) {
            const imgData = autoCapturedFrames[i].split(',')[1];
            
            const response = await fetch(`${API_BASE}/library/person`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    image: imgData,
                    notes: notes || undefined
                })
            });
            
            const data = await response.json();
            if (data.success) {
                successCount++;
            }
        }
        
        if (successCount > 0) {
            showToast(`Saved ${successCount} images for "${name}"`, 'success');
            logToTerminal(`> Saved ${successCount} images to library for ${name}`, 'success');
            loadLibrary();
            closeCaptureGalleryModal();
            autoCapturedFrames = [];
        } else {
            showToast('Failed to save images', 'error');
        }
        
    } catch (err) {
        logToTerminal(`> Error saving: ${err.message}`, 'error');
        showToast('Error saving: ' + err.message, 'error');
    }
    
    hideLoading();
}

// Face Mesh Overlay Functions
async function initFaceMesh() {
    faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    
    faceMesh.setOptions({
        maxNumFaces: 5,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    faceMesh.onResults(onMeshResults);
}

function onMeshResults(results) {
    const canvas = document.getElementById('meshCanvas');
    const ctx = canvas.getContext('2d');
    const video = document.getElementById('webcamVideo');
    
    if (!video.videoWidth || !video.videoHeight) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        for (const landmarks of results.multiFaceLandmarks) {
            drawMesh(ctx, landmarks, canvas.width, canvas.height);
        }
    }
}

function drawMesh(ctx, landmarks, width, height) {
    const connections = FaceMesh.FACE_CONNECTIONS || [];
    
    // Fallback: draw basic face contour if no connections
    if (connections.length === 0) {
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)';
        ctx.lineWidth = 1;
        
        // Draw simple face oval
        ctx.beginPath();
        ctx.ellipse(width/2, height/2, width*0.4, height*0.5, 0, 0, 2 * Math.PI);
        ctx.stroke();
    }
    
    ctx.lineWidth = 1;
    
    for (const [i, j] of connections) {
        if (i < landmarks.length && j < landmarks.length) {
            const p1 = landmarks[i];
            const p2 = landmarks[j];
            
            const z1 = p1.z || 0;
            const z2 = p2.z || 0;
            const avgZ = (z1 + z2) / 2;
            
            const intensity = Math.max(0, Math.min(255, Math.floor(128 + avgZ * 200)));
            const r = Math.floor(intensity / 2);
            const g = intensity;
            const b = 255 - intensity;
            
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        }
    }
    
    const keyPoints = [4, 10, 33, 133, 362, 263, 13, 82, 178, 400, 152, 234, 454];
    
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const x = landmark.x * width;
        const y = landmark.y * height;
        
        if (keyPoints.includes(i)) {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(0, 255, 255, 1)';
            ctx.fill();
        } else {
            ctx.beginPath();
            ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
            ctx.fill();
        }
    }
    
    ctx.font = '14px sans-serif';
    ctx.fillStyle = 'rgba(0, 255, 255, 1)';
    ctx.fillText('3D Face Mesh (MediaPipe - 478 points)', 10, 20);
}

async function toggleMeshOverlay() {
    const btn = document.getElementById('toggleMeshBtn');
    const canvas = document.getElementById('meshCanvas');
    const video = document.getElementById('webcamVideo');
    
    if (!meshOverlayActive) {
        if (!webcamStream) {
            showToast('Start webcam first', 'warning');
            return;
        }
        
        if (!faceMesh) {
            await initFaceMesh();
        }
        
        // Wait for video to be ready
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            showToast('Webcam not ready', 'warning');
            return;
        }
        
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Position canvas over video
        const container = document.getElementById('webcamContainer');
        canvas.style.width = video.offsetWidth + 'px';
        canvas.style.height = video.offsetHeight + 'px';
        
        meshOverlayActive = true;
        canvas.classList.add('active');
        btn.textContent = 'Hide Mesh';
        btn.disabled = false;
        
        // Start mesh processing loop
        processWebcamFrame();
        
        logToTerminal('> Mesh overlay enabled', 'success');
    } else {
        meshOverlayActive = false;
        if (meshCamera) {
            meshCamera.stop();
            meshCamera = null;
        }
        canvas.classList.remove('active');
        btn.textContent = 'Show Mesh';
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        logToTerminal('> Mesh overlay disabled', 'info');
    }
}

async function processWebcamFrame() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('meshCanvas');
    
    if (!meshOverlayActive || !video || video.paused || video.ended) {
        return;
    }
    
    if (faceMesh && meshOverlayActive) {
        try {
            await faceMesh.send({image: video});
        } catch (e) {
            // Ignore processing errors
        }
    }
    
    // Continue loop
    if (meshOverlayActive) {
        requestAnimationFrame(processWebcamFrame);
    }
}

// Switcher/Tab Previous Value Tracker for Liquid Glass Animations
const trackPrevious = (el) => {
    const radios = el.querySelectorAll('input[type="radio"]');
    let previousValue = null;

    const initiallyChecked = el.querySelector('input[type="radio"]:checked');
    if (initiallyChecked) {
        previousValue = initiallyChecked.getAttribute("c-option");
        el.setAttribute("c-previous", previousValue);
    }

    radios.forEach((radio) => {
        radio.addEventListener("change", () => {
            if (radio.checked) {
                el.setAttribute("c-previous", previousValue ?? "");
                previousValue = radio.getAttribute("c-option");
            }
        });
    });
};

// Auto-initialize switchers when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const switcher = document.querySelector(".switcher");
    if (switcher) {
        trackPrevious(switcher);
        
        // Theme switching functionality
        const radios = switcher.querySelectorAll('input[name="theme"]');
        radios.forEach(radio => {
            radio.addEventListener('change', () => {
                const theme = radio.value;
                document.body.setAttribute('data-theme', theme);
                logToTerminal(`> Theme changed to: ${theme}`, 'info');
            });
        });
    }
});

// ==========================================================================
// SIDEBAR FUNCTIONS
// ==========================================================================

let sidebarOpen = false;

function toggleSidebar() {
    sidebarOpen = !sidebarOpen;
    const sidebar = document.getElementById('sidebar');
    const container = document.querySelector('.container');
    const titlebarLeft = document.querySelector('.titlebar-left');
    
    if (sidebarOpen) {
        sidebar.classList.add('open');
        container.classList.add('sidebar-open');
        titlebarLeft.classList.add('sidebar-open');
        logToTerminal('> Sidebar opened', 'info');
    } else {
        sidebar.classList.remove('open');
        container.classList.remove('sidebar-open');
        titlebarLeft.classList.remove('sidebar-open');
        logToTerminal('> Sidebar closed', 'info');
    }
}

function jumpToStep(stepNum) {
    // Highlight in sidebar
    document.querySelectorAll('.sidebar-step').forEach(el => el.classList.remove('active'));
    const stepEl = document.getElementById(`sidebar-step-${stepNum}`);
    if (stepEl) {
        stepEl.classList.add('active');
    }
    
    // Highlight in titlebar
    document.querySelectorAll('.step-nav-icon').forEach(el => el.classList.remove('active'));
    const navIcons = document.querySelectorAll('.step-nav-icon');
    if (navIcons[stepNum - 1]) {
        navIcons[stepNum - 1].classList.add('active');
    }
    
    // Scroll to step in main content
    const stepElement = document.getElementById(`step${stepNum}`);
    if (stepElement) {
        stepElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        logToTerminal(`> Jumped to Step ${stepNum}`, 'info');
    }
}

// Scroll to section with active highlight
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        // Remove active-section from all steps
        document.querySelectorAll('.step').forEach(el => el.classList.remove('active-section'));
        // Add active-section to target
        section.classList.add('active-section');
        // Scroll with offset
        section.scrollIntoView({behavior: 'smooth', block: 'start'});
        logToTerminal(`> Scrolled to ${sectionId}`, 'info');
    }
}

// Traffic light handlers (Electron only)
function closeWindow() {
    if (window.electronAPI) {
        window.electronAPI.close();
    } else {
        logToTerminal('> Close window (Electron only)', 'info');
    }
}

function minimizeWindow() {
    if (window.electronAPI) {
        window.electronAPI.minimize();
    } else {
        logToTerminal('> Minimize window (Electron only)', 'info');
    }
}

function maximizeWindow() {
    if (window.electronAPI) {
        window.electronAPI.maximize();
    } else {
        logToTerminal('> Maximize window (Electron only)', 'info');
    }
}

// =============================================================================
// REFERENCE LIBRARY FUNCTIONS
// =============================================================================

let libraryPersons = [];
let libraryImageData = null;
let librarySource = 'upload';

// Show capture actions after webcam capture
function showCaptureActions() {
    const el = document.getElementById('captureActions');
    if (el) el.classList.remove('hidden');
}

function hideCaptureActions() {
    const el = document.getElementById('captureActions');
    if (el) el.classList.add('hidden');
}

// Use current image for matching (existing behavior)
function useForMatching() {
    hideCaptureActions();
    logToTerminal('> Use for matching', 'info');
}

// Show library modal
function showLibraryModal() {
    const modal = document.getElementById('libraryModal');
    const preview = document.getElementById('libraryPreviewImg');
    
    if (!modal) {
        console.warn('Library modal not found');
        return;
    }
    
    if (preview && currentImage) {
        preview.src = currentImage;
        preview.style.display = 'block';
        libraryImageData = currentImage;
        librarySource = currentWebcamActive ? 'webcam' : 'upload';
    } else if (preview) {
        preview.style.display = 'none';
    }
    
    modal.classList.remove('hidden');
    
    const nameInput = document.getElementById('libraryPersonName');
    if (nameInput) nameInput.focus();
}

// Close library modal
function closeLibraryModal() {
    const modal = document.getElementById('libraryModal');
    modal.classList.add('hidden');
    document.getElementById('libraryPersonName').value = '';
    document.getElementById('libraryPersonNotes').value = '';
    libraryImageData = null;
}

// Save to library
async function saveToLibrary() {
    const name = document.getElementById('libraryPersonName').value.trim();
    const notes = document.getElementById('libraryPersonNotes').value.trim();
    
    if (!name) {
        showToast('Please enter a name', 'warning');
        return;
    }
    
    if (!libraryImageData) {
        showToast('No image to save', 'warning');
        return;
    }
    
    showLoading('Saving to library...');
    
    try {
        const response = await fetch(`${API_BASE}/library/person`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                notes: notes,
                image: libraryImageData,
                source: librarySource
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`Saved "${name}" to library`, 'success');
            closeLibraryModal();
            hideCaptureActions();
            loadLibrary();
        } else {
            showToast(data.error || 'Failed to save', 'error');
        }
    } catch (err) {
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// Load library on startup
async function loadLibrary() {
    try {
        const response = await fetch(`${API_BASE}/library`);
        const data = await response.json();
        
        if (data.persons) {
            libraryPersons = data.persons;
            renderLibraryGrid();
            const statusEl = document.getElementById('libraryStatus');
            if (statusEl) {
                statusEl.textContent = `${data.count} person(s) in library`;
            }
        }
    } catch (err) {
        console.error('Failed to load library:', err);
        const statusEl = document.getElementById('libraryStatus');
        if (statusEl) {
            statusEl.textContent = 'Error loading library';
        }
    }
}

// Render library grid
function renderLibraryGrid() {
    const grid = document.getElementById('libraryGrid');
    
    if (!grid) {
        console.warn('Library grid element not found');
        return;
    }
    
    // Ensure grid is active/visible in normal grid mode
    grid.classList.add('active');
    grid.classList.remove('hidden');
    grid.classList.remove('results-mode');
    
    if (!libraryPersons.length) {
        grid.innerHTML = '<div class="empty-state">No persons in library. Add your first reference!</div>';
        return;
    }
    
    grid.innerHTML = libraryPersons.map(person => {
        const thumbnail = person.first_image_thumbnail || '';
        const personId = person.id;
        
        return `
        <div class="library-card" data-person-id="${personId}">
            <div class="library-card-image-container">
                <img src="${thumbnail}" class="library-card-thumb" alt="${person.name}">
            </div>
            <div class="library-card-content">
                <div class="library-card-name">${person.name}</div>
                <div class="library-card-info">${person.image_count} image(s)</div>
            </div>
            <div class="library-card-actions">
                <button class="btn-icon-small" onclick="event.stopPropagation(); viewLibraryPerson('${personId}')" title="View Info">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                </button>
                <button class="btn-icon-small btn-icon-danger" onclick="event.stopPropagation(); deleteLibraryPerson('${personId}')" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
        </div>
    `}).join('');
    
    // Update Step 4 library list as well
    loadStep4Library();
}

// View person details in popup
async function viewLibraryPerson(personId) {
    try {
        const response = await fetch(`${API_BASE}/library/person/${personId}`);
        const data = await response.json();
        
        if (data.success) {
            showLibraryInfoPopup(data.person, data.embeddings);
        }
    } catch (err) {
        showToast('Error loading person', 'error');
    }
}

// Show library info popup with meta info
function showLibraryInfoPopup(person, embeddings) {
    const popup = document.getElementById('libraryInfoPopup');
    const overlay = document.getElementById('libraryPopupOverlay');
    const nameEl = document.getElementById('popupPersonName');
    const contentEl = document.getElementById('popupContent');
    
    if (!popup || !overlay || !nameEl || !contentEl) return;
    
    // Set name
    nameEl.textContent = person.name;
    
    // Get first image data
    const firstImage = embeddings && embeddings.images && embeddings.images[0] ? embeddings.images[0] : null;
    
    // Build meta info HTML
    let metaHTML = '';
    
    // Basic info
    metaHTML += `
        <div class="meta-item">
            <span class="meta-label">ID</span>
            <span class="meta-value">${person.id}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Images</span>
            <span class="meta-value">${person.image_count}</span>
        </div>
    `;
    
    // Dates
    if (person.created_at) {
        const createdDate = new Date(person.created_at).toLocaleDateString();
        metaHTML += `
            <div class="meta-item">
                <span class="meta-label">Created</span>
                <span class="meta-value">${createdDate}</span>
            </div>
        `;
    }
    
    // Notes
    if (person.notes) {
        metaHTML += `
            <div class="meta-item">
                <span class="meta-label">Notes</span>
                <span class="meta-value">${person.notes}</span>
            </div>
        `;
    }
    
    // Image details if available
    if (firstImage) {
        if (firstImage.pose_category) {
            metaHTML += `
                <div class="meta-item">
                    <span class="meta-label">Pose</span>
                    <span class="meta-value">${firstImage.pose_category}</span>
                </div>
            `;
        }
        
        if (firstImage.quality) {
            const quality = firstImage.quality;
            const blurScore = quality.blur_score ? (quality.blur_score * 100).toFixed(1) + '%' : 'N/A';
            const brightness = quality.brightness ? (quality.brightness * 100).toFixed(1) + '%' : 'N/A';
            metaHTML += `
                <div class="meta-item">
                    <span class="meta-label">Quality</span>
                    <span class="meta-value">Sharp: ${blurScore}, Bright: ${brightness}</span>
                </div>
            `;
        }
        
        // Add thumbnail
        if (firstImage.thumbnail) {
            metaHTML += `<img src="data:image/jpeg;base64,${firstImage.thumbnail}" class="meta-image" alt="${person.name}">`;
        }
    }
    
    contentEl.innerHTML = metaHTML;
    
    // Show popup
    popup.classList.add('active');
    overlay.classList.add('active');
}

// Close library info popup
function closeLibraryInfoPopup() {
    const popup = document.getElementById('libraryInfoPopup');
    const overlay = document.getElementById('libraryPopupOverlay');
    
    if (popup) popup.classList.remove('active');
    if (overlay) overlay.classList.remove('active');
}

// Delete person
async function deleteLibraryPerson(personId) {
    if (!confirm('Delete this person from library?')) return;
    
    try {
        console.log('[DELETE LIBRARY] Deleting person:', personId);
        const response = await fetch(`${API_BASE}/library/person/${personId}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        console.log('[DELETE LIBRARY] Response:', data);
        
        if (data.success) {
            showToast('Person deleted from library', 'success');
            
            // Close popup if open
            const popup = document.getElementById('libraryInfoPopup');
            const overlay = document.getElementById('libraryPopupOverlay');
            if (popup) popup.classList.remove('active');
            if (overlay) overlay.classList.remove('active');
            
            loadLibrary();
        } else {
            showToast(data.error || 'Error deleting person', 'error');
        }
    } catch (err) {
        console.error('[DELETE LIBRARY] Exception:', err);
        showToast('Error: ' + err.message, 'error');
    }
}

// Handle library upload
function handleLibraryUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        libraryImageData = e.target.result;
        librarySource = 'upload';
        showLibraryModal();
    };
    reader.readAsDataURL(file);
    event.target.value = '';
}

// Start webcam for library
let currentWebcamActive = false;

async function startWebcamForLibrary() {
    currentWebcamActive = true;
    await startWebcam();
    
    // Scroll to webcam section
    const webcamStep = document.getElementById('webcamStep');
    if (webcamStep) {
        webcamStep.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    currentWebcamActive = false;
}

// Compare with library from Step 1 - scroll to compare section first
async function compareWithUploadedImage() {
    scrollToSection('step4');
    // Wait a bit for scroll, then compare
    setTimeout(() => {
        compareWithLibrary();
    }, 300);
}

// Add currently uploaded image to library
function addUploadedToLibrary() {
    if (!currentImage) {
        showToast('No image uploaded', 'warning');
        return;
    }
    
    // Use current image as library image
    libraryImageData = currentImage;
    librarySource = 'upload';
    
    // Show library modal with the current image
    showLibraryModal();
}

// Compare with library
async function compareWithLibrary() {
    if (!currentImage) {
        showToast('No image to compare', 'warning');
        return;
    }
    
    // Clear previous results
    clearComparisonResults();
    
    showLoading('Comparing with library...');
    logToTerminal('> Comparing with library...', 'command');
    
    try {
        const response = await fetch(`${API_BASE}/library/match`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImage })
        });
        
        const data = await response.json();
        
        if (data.success && data.matches.length > 0) {
            const best = data.matches[0];
            logToTerminal(`> Best match: ${best.person_name} (${(best.score * 100).toFixed(1)}%)`, 'success');
            
            // Update comparison result display
            const matchStatusEl = document.getElementById('matchStatus');
            const matchScoreEl = document.getElementById('matchScore');
            const comparisonResultEl = document.getElementById('comparisonResult');
            
            // Show name only if score >= 60%
            const scorePercent = best.score * 100;
            if (scorePercent >= 60) {
                if (matchStatusEl) matchStatusEl.textContent = best.person_name;
                showToast(`Best match: ${best.person_name}`, 'success');
            } else {
                if (matchStatusEl) matchStatusEl.textContent = 'No Match';
                showToast('No match found (below 60%)', 'info');
            }
            
            if (matchScoreEl) matchScoreEl.textContent = `${scorePercent.toFixed(1)}%`;
            if (comparisonResultEl) {
                comparisonResultEl.classList.remove('hidden');
                comparisonResultEl.classList.add('visible');
                comparisonResultEl.classList.add('active');
                comparisonResultEl.style.display = 'flex';
            }
            
            // Show images in comparison result
            const queryImageEl = document.getElementById('queryImage');
            const refImageEl = document.getElementById('refImage');
            const refLabelEl = document.getElementById('refLabel');
            
            // Show query image
            if (queryImageEl && currentImage) {
                queryImageEl.src = currentImage;
                queryImageEl.style.display = 'inline-block';
            }
            
            // Show reference image
            if (refImageEl && best.best_image && best.best_image.thumbnail) {
                refImageEl.src = `data:image/jpeg;base64,${best.best_image.thumbnail}`;
                refImageEl.style.display = 'inline-block';
            }
            if (refLabelEl) {
                refLabelEl.textContent = best.person_name;
            }
            
            // Auto-expand scores dropdown
            expandScoresDropdown();
            
            // Display detailed scores from library match
            const arcfaceEl = document.getElementById('arcfaceScore');
            if (arcfaceEl && best.arcface_similarity != null) {
                arcfaceEl.textContent = `${Math.round(best.arcface_similarity * 100)}%`;
            } else if (arcfaceEl) {
                arcfaceEl.textContent = '--%';
            }
            
            const facenetEl = document.getElementById('facenetScore');
            if (facenetEl && best.facenet_similarity != null) {
                facenetEl.textContent = `${Math.round(best.facenet_similarity * 100)}%`;
            } else if (facenetEl) {
                facenetEl.textContent = '--%';
            }
            
            // Display all other scores
            const normEl = document.getElementById('normScore');
            if (normEl && best.normalized_similarity != null) {
                normEl.textContent = `${Math.round(best.normalized_similarity * 100)}%`;
            } else if (normEl) {
                normEl.textContent = '--%';
            }
            
            const multiPoseEl = document.getElementById('multiPoseScore');
            if (multiPoseEl && best.multi_pose_score != null) {
                multiPoseEl.textContent = `${Math.round(best.multi_pose_score * 100)}%`;
            } else if (multiPoseEl) {
                multiPoseEl.textContent = '--%';
            }
            
            const textureEl = document.getElementById('textureScore');
            if (textureEl && best.texture_similarity != null) {
                textureEl.textContent = `${Math.round(best.texture_similarity * 100)}%`;
            } else if (textureEl && best.lbp_similarity != null) {
                textureEl.textContent = `${Math.round(best.lbp_similarity * 100)}%`;
            } else if (textureEl) {
                textureEl.textContent = '--%';
            }
            
            const uniquenessEl = document.getElementById('uniquenessScore');
            if (uniquenessEl && best.uniqueness_similarity != null) {
                uniquenessEl.textContent = `${Math.round(best.uniqueness_similarity * 100)}%`;
            } else if (uniquenessEl && best.asymmetry_similarity != null) {
                uniquenessEl.textContent = `${Math.round(best.asymmetry_similarity * 100)}%`;
            } else if (uniquenessEl) {
                uniquenessEl.textContent = '--%';
            }
            
            const activationEl = document.getElementById('activationScore');
            if (activationEl && best.activation_similarity != null) {
                activationEl.textContent = `${Math.round(best.activation_similarity * 100)}%`;
            } else if (activationEl) {
                activationEl.textContent = '--%';
            }
            
            // Display Iris score
            const irisEl = document.getElementById('irisScore');
            if (irisEl && best.iris_similarity != null) {
                irisEl.textContent = `${Math.round(best.iris_similarity * 100)}%`;
            } else if (irisEl) {
                irisEl.textContent = '--%';
            }
            
            // Display Expression score
            const exprEl = document.getElementById('expressionScore');
            if (exprEl && best.expression_similarity != null) {
                exprEl.textContent = `${Math.round(best.expression_similarity * 100)}%`;
            } else if (exprEl) {
                exprEl.textContent = '--%';
            }
            
            // Show thumbnail if available
            if (best.best_image && best.best_image.thumbnail) {
                document.getElementById('refImage').src = `data:image/jpeg;base64,${best.best_image.thumbnail}`;
            }
        } else if (data.success && data.matches.length === 0) {
            logToTerminal('> No matches found in library', 'info');
            showToast('No matches found', 'info');
        } else {
            logToTerminal(`> Error: ${data.error}`, 'error');
            showToast(data.error || 'Match failed', 'error');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// Enable library compare button when we have an image
function checkLibraryCompareButton() {
    const btn = document.getElementById('compareLibraryBtn');
    if (btn) {
        btn.disabled = !currentImage;
    }
}

function checkFindMatchesButton() {
    const btn = document.getElementById('findMatchesBtn');
    if (btn) {
        btn.disabled = !currentImage;
    }
}

// Search library by name
async function searchLibraryByName(name) {
    if (!name || name.length < 1) {
        renderLibraryGrid();
        return;
    }
    
    const searchTerm = name.toLowerCase();
    const found = libraryPersons.filter(p => 
        p.name.toLowerCase().includes(searchTerm)
    );
    
    const grid = document.getElementById('libraryGrid');
    if (!grid) {
        console.warn('Library grid element not found');
        return;
    }
    
    if (found.length === 0) {
        grid.innerHTML = '<div class="empty-state">No persons found matching "' + name + '"</div>';
    } else {
        grid.innerHTML = found.map(person => `
            <div class="library-card" onclick="viewLibraryPerson('${person.id}')">
                <div class="library-card-name">${person.name}</div>
                <div class="library-card-info">${person.image_count} image(s)</div>
                <button class="btn-delete" onclick="event.stopPropagation(); deleteLibraryPerson('${person.id}')">Delete</button>
            </div>
        `).join('');
    }
}

// Match current image with library - uses comparison-result style
async function matchWithLibraryImage(imageData) {
    if (!imageData) {
        showToast('No image to match', 'warning');
        return;
    }
    
    showLoading('Matching with library...');
    
    try {
        const response = await fetch(`${API_BASE}/library/match`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        const grid = document.getElementById('libraryGrid');
        
        if (!grid) {
            console.warn('Library grid element not found');
            return;
        }
        
        // Switch to results mode (flex column)
        grid.classList.add('results-mode');
        
        if (data.success && data.matches.length > 0) {
            logToTerminal(`> Found ${data.matches.length} match(es) in library`, 'success');
            
            // Use comparison-result style for matches
            grid.innerHTML = `
                <div class="library-matches-header">
                    <h3>Best Matches</h3>
                    <button class="btn" onclick="renderLibraryGrid()">Back to Library</button>
                </div>
                <div class="library-matches-grid">
                    ${data.matches.map((match, index) => `
                        <div class="library-match-card ${index === 0 ? 'best-match' : ''}" onclick="viewLibraryPerson('${match.person_id}')">
                            <div class="match-rank">#${index + 1}</div>
                            <div class="match-thumbnail">
                                ${match.best_image && match.best_image.thumbnail ? 
                                    `<img src="data:image/jpeg;base64,${match.best_image.thumbnail}" alt="${match.person_name}">` : 
                                    '<div class="no-thumb">👤</div>'}
                            </div>
                            <div class="match-info">
                                <div class="match-name">${match.person_name}</div>
                                <div class="match-score ${getScoreClass(match.score)}">${(match.score * 100).toFixed(1)}%</div>
                                <div class="match-label">${getMatchLabel(match.score)}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            showToast(`Found ${data.matches.length} match(es)!`, 'success');
        } else {
            grid.innerHTML = '<div class="empty-state">No matches found in library</div>';
            showToast('No matches found', 'info');
        }
    } catch (err) {
        console.error('Match error:', err);
        showToast('Error: ' + err.message, 'error');
        renderLibraryGrid();
    } finally {
        hideLoading();
    }
}

// Helper functions for match display
function getScoreClass(score) {
    if (score >= 0.7) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
}

function getMatchLabel(score) {
    if (score >= 0.7) return 'Strong Match';
    if (score >= 0.5) return 'Possible Match';
    return 'Weak Match';
}

// Handle upload specifically for comparing (not adding to library)
function handleLibraryCompareUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const imageData = e.target.result;
        matchWithLibraryImage(imageData);
    };
    reader.readAsDataURL(file);
    event.target.value = ''; // Reset so same file can be selected again
}

// Legacy function - redirects to correct handler
async function searchLibrary(query) {
    if (!query) {
        renderLibraryGrid();
        return;
    }
    if (typeof query === 'string' && query.startsWith('data:image')) {
        await matchWithLibraryImage(query);
    } else {
        await searchLibraryByName(query);
    }
}

// Load library on startup
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(loadLibrary, 1000);
});
