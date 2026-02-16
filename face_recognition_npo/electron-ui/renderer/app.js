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
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            const vizType = e.target.dataset.viz;
            logToTerminal(`>>> CLICKED TAB: ${vizType}`, 'info');
            showVisualization(vizType);
        });
    });
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
    
    tabsEl.innerHTML = tabs.map(t => 
        `<div class="ref-viz-tab ${t.id === 'info' ? 'active' : ''}" data-tab="${t.id}" onclick="switchRefTab('${t.id}', ${refId})">${t.label}</div>`
    ).join('');
    
    switchRefTab('info', refId);
}

async function switchRefTab(tabId, refId) {
    const ref = references[refId];
    const tabs = document.querySelectorAll('.ref-viz-tab');
    tabs.forEach(t => t.classList.remove('active'));
    document.querySelector(`.ref-viz-tab[data-tab="${tabId}"]`)?.classList.add('active');
    
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
                reasonsEl.innerHTML = '<ul>' + best.reasons.map(r => `<li>${r}</li>`).join('') + '</ul>';
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
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        webcamStream = stream;
        video.srcObject = stream;
        container.classList.add('visible');
        
        startBtn.disabled = true;
        captureBtn.disabled = false;
        stopBtn.disabled = false;
        
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
    container.classList.add('hidden');
    
    startBtn.disabled = false;
    captureBtn.disabled = true;
    stopBtn.disabled = true;
    
    status.textContent = 'Webcam stopped';
    status.className = 'status';
    logToTerminal('> Webcam stopped', 'info');
}
