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
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('facesContainer').style.display = 'none';
    document.getElementById('comparisonResult').classList.remove('active');
    document.getElementById('referenceList').innerHTML = '';
    
    document.getElementById('detectStatus').textContent = 'Waiting for image...';
    document.getElementById('detectStatus').className = 'status status-info';
    document.getElementById('extractStatus').textContent = 'Waiting for detection...';
    document.getElementById('extractStatus').className = 'status status-info';
    document.getElementById('compareStatus').textContent = 'Step 1: Detect faces first';
    document.getElementById('compareStatus').className = 'status status-info';
    
    showVisualizationPlaceholder();
    clearTerminal();
    logToTerminal('> Cache cleared', 'success');
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
        document.getElementById('selectedImage').src = currentImage;
        document.getElementById('previewContainer').style.display = 'flex';
        document.getElementById('detectBtn').disabled = false;
        document.getElementById('detectStatus').textContent = 'Ready to detect';
        document.getElementById('detectStatus').className = 'status status-info';
        resetSteps();
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
    document.getElementById('facesContainer').style.display = 'none';
    document.getElementById('extractBtn').disabled = true;
    document.getElementById('extractStatus').textContent = 'Waiting for detection...';
    document.getElementById('compareStatus').textContent = 'Step 1: Detect faces first';
    document.getElementById('compareBtn').disabled = true;
    document.getElementById('comparisonResult').style.display = 'none';
    visualizationData = {};
    showVisualizationPlaceholder();
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

            if (currentQueryEmbedding !== null) {
                await compareFaces();
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

            document.getElementById('facesContainer').style.display = 'block';

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
            document.getElementById('compareBtn').disabled = false;
            document.getElementById('compareStatus').textContent = 'Step 4: Click "Compare" to find matches';

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
                <p style="color: #cc0000;">${err.message}</p>
            </div>
        `;
    }
}

function updateReferenceList() {
    const container = document.getElementById('referenceList');
    container.innerHTML = '';
    
    if (!references || references.length === 0) {
        container.innerHTML = '<p style="color: #666; padding: 8px;">No references added yet</p>';
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
            logToTerminal(`> Similarity score: ${(best.similarity * 100).toFixed(1)}%`, 'success');
            logToTerminal(`> Confidence: ${best.confidence}`, 'info');

            document.getElementById('queryImage').src = `data:image/png;base64,${currentFaceThumbnails[0]?.thumbnail || ''}`;
            document.getElementById('refImage').src = `data:image/png;base64,${best.thumbnail}`;
            document.getElementById('refLabel').textContent = best.name;
            document.getElementById('matchScore').textContent = `${Math.round(best.similarity * 100)}%`;
            document.getElementById('confidenceBadge').textContent = best.confidence;

            // Set badge color
            const badge = document.getElementById('confidenceBadge');
            if (best.similarity > 0.8) {
                badge.className = 'badge badge-success';
            } else if (best.similarity > 0.6) {
                badge.className = 'badge badge-warning';
            } else {
                badge.className = 'badge badge-error';
            }

            document.getElementById('comparisonResult').style.display = 'block';
            document.getElementById('compareStatus').textContent = `Best match: ${best.name} (${Math.round(best.similarity * 100)}%)`;
            document.getElementById('compareStatus').className = 'status status-success';

            // Store similarity visualization
            visualizationData['similarity'] = data.similarity_viz;
            visualizationData['similarity_data'] = data.similarity_data;
            showVisualization('similarity');

            showToast(`Match: ${best.name} (${Math.round(best.similarity * 100)}%)`, 'success');
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
                <p style="color: #e74c3c;">No face detected</p>
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
                <p style="color: #e74c3c;">No embedding extracted</p>
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

            if (data.success && data.visualization && data.visualization.length > 100) {
                visualizationData[vizType] = data.visualization;
                if (data.data && Object.keys(data.data).length > 0) {
                    visualizationData[vizType + '_data'] = data.data;
                }
                logToTerminal(`> Received ${data.visualization.length} chars for ${vizType}`, 'success');
            } else {
                console.log('[VIZ] API returned no/invalid data:', data);
                content.innerHTML = `
                    <div class="viz-placeholder">
                        <p style="color: #e74c3c;">No ${vizType} data available</p>
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
                    <p style="color: #e74c3c;">Error loading ${vizType}</p>
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

        let html = `<img src="data:image/png;base64,${visualizationData[vizType]}" alt="${vizType}" style="max-width: 100%; max-height: 400px; display: block; margin: 0 auto;">`;

        // Add data table if available
        const dataKey = vizType + '_data';
        if (visualizationData[dataKey]) {
            html += formatDataAsTable(visualizationData[dataKey]);
        }

        content.innerHTML = html;
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
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
