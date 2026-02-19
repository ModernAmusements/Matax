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
    
    document.getElementById('selectedImage').src = '';
    document.getElementById('previewContainer').classList.remove('visible');
    document.getElementById('previewContainer').classList.add('hidden');
    document.getElementById('step1').classList.remove('step-complete');
    document.getElementById('step2').classList.remove('step-complete');
    document.getElementById('step3').classList.remove('step-complete');
    document.getElementById('step4').classList.remove('step-complete');
    document.getElementById('webcamStep').classList.remove('step-complete');
    document.getElementById('detectBtn').classList.remove('btn-success');
    document.getElementById('detectBtn').classList.add('btn-primary');
    document.getElementById('extractBtn').classList.remove('btn-success');
    document.getElementById('extractBtn').classList.add('btn-primary');
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
        document.getElementById('selectedImage').src = currentImage;
        document.getElementById('previewContainer').classList.add('visible');
        document.getElementById('detectBtn').disabled = false;
        document.getElementById('detectStatus').textContent = 'Ready to detect';
        document.getElementById('detectStatus').className = 'status status-info';
        resetSteps();
        markStepComplete('step1', 'detectBtn');
        event.target.value = '';
        checkLibraryCompareButton();
        checkFindMatchesButton();
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
    document.getElementById('facesContainer').classList.add('hidden');
    document.getElementById('extractBtn').disabled = true;
    document.getElementById('extractStatus').textContent = 'Waiting for detection...';
    document.getElementById('compareStatus').textContent = 'Step 1: Detect faces first';
    document.getElementById('compareBtn').disabled = true;
    document.getElementById('comparisonResult').classList.add('hidden');
    visualizationData = {};
    showVisualizationPlaceholder();
    
    // Reset step states
    document.getElementById('step1').classList.remove('step-complete');
    document.getElementById('step2').classList.remove('step-complete');
    document.getElementById('step3').classList.remove('step-complete');
    document.getElementById('step4').classList.remove('step-complete');
    document.getElementById('webcamStep').classList.remove('step-complete');
    
    // Reset button states
    document.getElementById('detectBtn').classList.remove('btn-success');
    document.getElementById('extractBtn').classList.remove('btn-success');
}

function markStepComplete(stepId, btnId) {
    document.getElementById(stepId).classList.add('step-complete');
    if (btnId && document.getElementById(btnId)) {
        document.getElementById(btnId).classList.remove('btn-primary');
        document.getElementById(btnId).classList.add('btn-success');
    }
}

function selectImage() {
    document.getElementById('imageInput').click();
}

function addReference() {
    document.getElementById('refInput').click();
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
            document.getElementById('detectStatus').textContent = `Found ${data.count} face(s)!`;
            document.getElementById('detectStatus').className = 'status status-success';
            document.getElementById('extractBtn').disabled = false;
            markStepComplete('step2', 'extractBtn');

            // Display preprocessing info
            if (data.preprocessing) {
                const prep = data.preprocessing;
                if (prep.was_enhanced) {
                    const msg = `Image enhanced: ${prep.method.toUpperCase()} (quality: ${(prep.enhanced_quality.overall * 100).toFixed(0)}%)`;
                    logToTerminal('> ' + msg, 'info');
                    document.getElementById('detectStatus').textContent = `Found ${data.count} face(s) - ${prep.method} enhanced`;
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
                    document.getElementById('detectStatus').textContent = `Found ${data.count} face(s) - ${ew.type} detected!`;
                    document.getElementById('detectStatus').className = 'status status-warning';
                    showToast(warningMsg, 'warning');
                }
            } catch (ewErr) {
                console.log('[EYEWEAR] Check failed:', ewErr.message);
            }

            const gallery = document.getElementById('facesGallery');
            gallery.innerHTML = '';
            currentFaceThumbnails = data.faces;

            data.faces.forEach((face, i) => {
                logToTerminal(`> Face ${i + 1}: bbox=[${face.bbox.join(', ')}]`, 'info');
                const div = document.createElement('div');
                div.className = 'gallery-item';
                div.innerHTML = `
                    <img src="data:image/png;base64,${face.thumbnail}" alt="Face ${i + 1}">
                    <span>Face ${i + 1}</span>
                `;
                gallery.appendChild(div);
            });

            document.getElementById('facesContainer').classList.add('visible');

            Object.keys(data.visualizations).forEach(key => {
                visualizationData[key] = data.visualizations[key];
            });

            showVisualization('detection');
            showToast(`Found ${data.count} face(s)`, 'success');
        } else {
            logToTerminal('> No faces detected', 'error');
            document.getElementById('detectStatus').textContent = 'No faces detected';
            document.getElementById('detectStatus').className = 'status status-warning';
            showToast(data.error || 'No faces detected', 'warning');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        document.getElementById('detectStatus').textContent = 'Error detecting faces';
        document.getElementById('detectStatus').className = 'status status-error';
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
            document.getElementById('extractStatus').textContent = `Features extracted (${data.embedding_size}-dim)`;
            document.getElementById('extractStatus').className = 'status status-success';
            markStepComplete('step3', null);
            
            // Enable compare button only if we have both embedding AND references
            const hasReferences = references && references.length > 0;
            document.getElementById('compareBtn').disabled = !hasReferences;
            if (hasReferences) {
                document.getElementById('compareStatus').textContent = 'Step 4: Click "Compare" to find matches';
            } else {
                document.getElementById('compareStatus').textContent = 'Step 3b: Add a reference image to compare';
            }

            console.log('[EXTRACT] Cached visualizations:', Object.keys(visualizationData));
            
            showVisualization('embedding');
            showToast('Features extracted successfully', 'success');
        } else {
            logToTerminal('> Feature extraction failed', 'error');
            document.getElementById('extractStatus').textContent = 'Extraction failed';
            document.getElementById('extractStatus').className = 'status status-error';
            showToast(data.error || 'Extraction failed', 'error');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        document.getElementById('extractStatus').textContent = 'Error extracting features';
        document.getElementById('extractStatus').className = 'status status-error';
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
        checkLibraryCompareButton();
        checkFindMatchesButton();
    }
}

async function removeReference(index, event) {
    if (event) {
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
    
    const btn = event?.target;
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
            logToTerminal(`> Removed: ${refName}`, 'success');
            showToast('Reference removed', 'success');
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (err) {
        logToTerminal(`> Error removing ${refName}: ${err.message}`, 'error');
        showToast('Failed to remove reference', 'error');
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

function showReferenceDetailsOnly(refIndex, event) {
    event.stopPropagation();
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
            <div class="ref-remove-btn" onclick="removeReference(${i}, event)">×</div>
            <div class="ref-details-btn" onclick="showReferenceDetailsOnly(${i}, event)" title="View Details">i</div>
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
    }
}

// Track selected library refs for comparison
let selectedLibraryRefs = [];

function selectLibraryRefForCompare(personId, personName) {
    // Toggle selection
    const index = selectedLibraryRefs.indexOf(personId);
    if (index > -1) {
        selectedLibraryRefs.splice(index, 1);
    } else {
        selectedLibraryRefs.push(personId);
    }
    
    // Update UI to show selection
    const items = document.querySelectorAll('.library-ref-item');
    items.forEach(item => {
        const onclickAttr = item.getAttribute('onclick');
        if (onclickAttr && onclickAttr.includes(`'${personId}'`)) {
            item.classList.toggle('selected');
        }
    });
    
    // Enable/disable compare button
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        compareBtn.disabled = selectedLibraryRefs.length === 0 && (!references || references.length === 0);
        if (selectedLibraryRefs.length > 0) {
            compareBtn.textContent = `Compare with Selected (${selectedLibraryRefs.length})`;
        } else {
            compareBtn.textContent = 'Compare with Selected';
        }
    }
    
    logToTerminal(`> Selected library ref: ${personName}`, 'info');
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

// Compare Faces
async function compareFaces() {
    logToTerminal(`> Compare: currentQueryEmbedding=${currentQueryEmbedding}, references.length=${references.length}`, 'info');

    if (currentQueryEmbedding === null) {
        logToTerminal('> Error: No embedding extracted. Please click \"Create Signature\" first.', 'error');
        showToast('Extract features first!', 'error');
        return;
    }
    if (references.length === 0) {
        logToTerminal('> Error: No references added. Add a reference image first.', 'error');
        showToast('Add at least one reference', 'warning');
        return;
    }

    showLoading('Comparing...');
    logToTerminal('> Initializing similarity comparison...', 'command');
    logToTerminal(`> Query embedding: ${currentQueryEmbedding?.toFixed(6) || 'null'}`, 'info');
    logToTerminal(`> Comparing against ${references.length} reference(s)...`, 'info');

    try {
        logToTerminal('> Computing cosine similarities...', 'info');
        const response = await fetch(`${API_BASE}/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

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

            document.getElementById('queryImage').src = `data:image/png;base64,${currentFaceThumbnails[0]?.thumbnail || ''}`;
            document.getElementById('refImage').src = `data:image/png;base64,${best.thumbnail}`;
            document.getElementById('refLabel').textContent = best.name;
            
            // Display match status
            const statusEl = document.getElementById('matchStatus');
            statusEl.textContent = best.match_label;
            statusEl.className = `comparison-status ${best.status}`;
            
            // Display ArcFace score
            const arcfaceEl = document.getElementById('arcfaceScore');
            if (best.arcface_similarity !== null && best.arcface_similarity !== undefined) {
                arcfaceEl.textContent = `${Math.round(best.arcface_similarity * 100)}%`;
            } else {
                arcfaceEl.textContent = 'N/A';
            }
            
            // Display FaceNet score
            const facenetEl = document.getElementById('facenetScore');
            if (best.facenet_similarity !== null && best.facenet_similarity !== undefined) {
                facenetEl.textContent = `${Math.round(best.facenet_similarity * 100)}%`;
            } else {
                facenetEl.textContent = 'N/A';
            }
            
            // Display Activation similarity score
            const activationEl = document.getElementById('activationScore');
            if (best.activation_similarity !== null && best.activation_similarity !== undefined) {
                activationEl.textContent = `${Math.round(best.activation_similarity * 100)}%`;
            } else {
                activationEl.textContent = 'N/A';
            }
            
            // Display 3D Normalized score
            const normEl = document.getElementById('normScore');
            if (best.normalized_similarity !== null && best.normalized_similarity !== undefined) {
                normEl.textContent = `${Math.round(best.normalized_similarity * 100)}%`;
            } else {
                normEl.textContent = 'N/A';
            }
            
            // Display Multi-Pose score
            const multiPoseEl = document.getElementById('multiPoseScore');
            if (best.multi_pose_score !== null && best.multi_pose_score !== undefined) {
                multiPoseEl.textContent = `${Math.round(best.multi_pose_score * 100)}%`;
            } else {
                multiPoseEl.textContent = 'N/A';
            }
            
            // Display Texture (LBP) score
            const lbpEl = document.getElementById('lbpScore');
            if (best.lbp_similarity !== null && best.lbp_similarity !== undefined) {
                lbpEl.textContent = `${Math.round(best.lbp_similarity * 100)}%`;
            } else {
                lbpEl.textContent = 'N/A';
            }
            
            // Display Uniqueness (Asymmetry) score
            const asymEl = document.getElementById('asymScore');
            if (best.asymmetry_similarity !== null && best.asymmetry_similarity !== undefined) {
                asymEl.textContent = `${Math.round(best.asymmetry_similarity * 100)}%`;
            } else {
                asymEl.textContent = 'N/A';
            }
            
            // Display final combined score
            document.getElementById('matchScore').textContent = `${Math.round(best.final_score * 100)}%`;
            
            // Display reasons
            const reasonsEl = document.getElementById('matchReasons');
            if (best.reasons && best.reasons.length > 0) {
                reasonsEl.innerHTML = `
                    <div class="match-reasons-toggle" onclick="this.setAttribute('data-expanded', this.getAttribute('data-expanded') === 'true' ? 'false' : 'true'); this.nextElementSibling.setAttribute('data-visible', this.getAttribute('data-expanded') === 'true' ? 'true' : 'false')">
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

            const comparisonResult = document.getElementById('comparisonResult');
            comparisonResult.classList.add('visible');
            comparisonResult.classList.add('active');
            document.getElementById('compareStatus').textContent = `Best match: ${best.name} (${Math.round(best.final_score * 100)}%)`;
            document.getElementById('compareStatus').className = 'status status-success';

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
            document.getElementById('compareStatus').textContent = errorMsg;
            document.getElementById('compareStatus').className = 'status status-warning';
            showToast(errorMsg, 'warning');
        }
    } catch (err) {
        logToTerminal(`> Error: ${err.message}`, 'error');
        document.getElementById('compareStatus').textContent = 'Error comparing';
        document.getElementById('compareStatus').className = 'status status-error';
        showToast('Error: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// Visualization
async function showVisualization(vizType) {
    const content = document.getElementById('vizContent');
    
    console.log('[VIZ] Requested:', vizType);
    
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
        
        startBtn.disabled = true;
        captureBtn.disabled = false;
        stopBtn.disabled = false;
        
        const toggleMeshBtn = document.getElementById('toggleMeshBtn');
        if (toggleMeshBtn) {
            toggleMeshBtn.disabled = false;
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
    
    status.textContent = 'Webcam stopped';
    status.className = 'status';
    logToTerminal('> Webcam stopped', 'info');
    
    if (meshOverlayActive) {
        toggleMeshOverlay();
    }
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
        const response = await fetch(`${API_BASE}/library/person/${personId}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        
        if (data.success) {
            showToast('Person deleted', 'success');
            loadLibrary();
        } else {
            showToast(data.error || 'Error deleting', 'error');
        }
    } catch (err) {
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

// Compare with library
async function compareWithLibrary() {
    if (!currentImage) {
        showToast('No image to compare', 'warning');
        return;
    }
    
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
            showToast(`Best match: ${best.person_name}`, 'success');
            
            // Update comparison result display
            document.getElementById('matchStatus').textContent = best.person_name;
            document.getElementById('matchScore').textContent = `${(best.score * 100).toFixed(1)}%`;
            
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
