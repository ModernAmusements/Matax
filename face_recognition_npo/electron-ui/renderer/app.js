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

function updateLoadingText(text) {
    document.getElementById('loadingText').textContent = text;
}

// Setup event listeners
function setupEventListeners() {
    // Visualization tabs
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            showVisualization(e.target.dataset.viz);
        });
    });
}

// Check API connection
async function checkAPI() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        if (data.status === 'ok') {
            console.log('API connected');
        }
    } catch (err) {
        showToast('Cannot connect to API server. Make sure api_server.py is running.', 'error');
    }
}

// Image Selection
function selectImage() {
    document.getElementById('imageInput').click();
}

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = e.target.result;
            document.getElementById('selectedImage').src = currentImage;
            document.getElementById('previewContainer').style.display = 'flex';
            document.getElementById('detectBtn').disabled = false;
            document.getElementById('detectStatus').textContent = 'Ready to detect';
            document.getElementById('detectStatus').className = 'status status-info';
            resetSteps();
        };
        reader.readAsDataURL(file);
    }
}

// Reset steps
function resetSteps() {
    currentFaceThumbnails = [];
    currentQueryEmbedding = null;
    document.getElementById('facesContainer').style.display = 'none';
    document.getElementById('extractBtn').disabled = true;
    document.getElementById('extractStatus').textContent = 'Waiting for detection...';
    document.getElementById('compareStatus').textContent = 'Waiting for extraction...';
    document.getElementById('comparisonResult').style.display = 'none';
    visualizationData = {};
    showVisualizationPlaceholder();
}

// Detect Faces
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

            // Show face thumbnails
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

            // Store visualization data
            Object.keys(data.visualizations).forEach(key => {
                visualizationData[key] = data.visualizations[key];
            });

            // Show first visualization
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

// Extract Features
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
            logToTerminal(`> Embedding vector: ${data.embedding_size} dimensions`, 'success');
            logToTerminal(`> Mean: ${data.embedding_mean.toFixed(6)}, Std: ${data.embedding_std.toFixed(6)}`, 'info');
            document.getElementById('extractStatus').textContent = `Features extracted (${data.embedding_size}-dim)`;
            document.getElementById('extractStatus').className = 'status status-success';
            document.getElementById('compareBtn').disabled = false;
            document.getElementById('compareStatus').textContent = 'Ready to compare';

            // Store visualization data
            Object.keys(data.visualizations).forEach(key => {
                visualizationData[key] = data.visualizations[key];
            });
            
            // Store visualization data tables (raw data)
            if (data.visualization_data) {
                Object.keys(data.visualization_data).forEach(key => {
                    visualizationData[key + '_data'] = data.visualization_data[key];
                });
            }
            
            // Show embedding visualization
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

// Add Reference
function addReference() {
    document.getElementById('refInput').click();
}

function handleReferenceSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            await saveReference(e.target.result, file.name);
        };
        reader.readAsDataURL(file);
    }
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
                // Auto compare
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

function updateReferenceList() {
    const container = document.getElementById('referenceList');
    container.innerHTML = '';

    references.forEach((ref, i) => {
        const div = document.createElement('div');
        div.className = 'reference-item';
        div.onclick = () => selectReference(i);
        div.innerHTML = `
            <img src="data:image/png;base64,${ref.thumbnail}" alt="${ref.name}">
            <span>${ref.name}</span>
        `;
        container.appendChild(div);
    });
}

function selectReference(index) {
    selectedReferenceId = index;
}

// Compare Faces
async function compareFaces() {
    if (currentQueryEmbedding === null || references.length === 0) {
        if (references.length === 0) {
            showToast('Add at least one reference', 'warning');
        }
        return;
    }

    showLoading('Comparing...');
    logToTerminal('> Initializing similarity comparison...', 'command');
    logToTerminal(`> Query embedding: ${currentQueryEmbedding.toFixed(6)}`, 'info');
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
            logToTerminal('> No match found', 'warning');
            document.getElementById('compareStatus').textContent = 'No match found';
            document.getElementById('compareStatus').className = 'status status-warning';
            showToast('No match found', 'warning');
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
    
    // If data not available locally, fetch from API
    if (!visualizationData[vizType]) {
        try {
            const response = await fetch(`${API_BASE}/visualizations/${vizType}`);
            const data = await response.json();
            if (data.success) {
                visualizationData[vizType] = data.visualization;
                if (data.data && Object.keys(data.data).length > 0) {
                    visualizationData[vizType + '_data'] = data.data;
                }
            }
        } catch (err) {
            console.error('Failed to fetch visualization:', err);
        }
    }
    
    if (visualizationData[vizType]) {
        let html = `<img src="data:image/png;base64,${visualizationData[vizType]}" alt="${vizType}" style="max-width: 100%; max-height: 400px; display: block; margin: 0 auto;">`;
        
        // Add data table if available
        const dataKey = vizType + '_data';
        if (visualizationData[dataKey]) {
            html += formatDataAsTable(visualizationData[dataKey]);
        }
        
        content.innerHTML = html;
    } else {
        showVisualizationPlaceholder();
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
