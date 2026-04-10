// ============================================================================
// FILE: 03-workflows.js
// PURPOSE: User action handlers - Upload, Webcam, Library, References
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. PUBLIC API - Image Upload
    // ============================================================================

    /**
     * Handle image file selection
     * @param {Event} event - File input change event
     */
    function handleImageSelect(event) {
        var file = event.target.files[0];
        
        if (!file) return;
        
        // Validate file type
        var validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        
        if (validTypes.indexOf(file.type) === -1) {
            UI.showToast('Please select an image file (JPEG, PNG, GIF, WebP)', 'error');
            event.target.value = '';
            return;
        }
        
        // Validate file size (10MB max)
        var maxSize = 10 * 1024 * 1024;
        
        if (file.size > maxSize) {
            UI.showToast('Image too large (max 10MB)', 'warning');
            event.target.value = '';
            return;
        }
        
        var reader = new FileReader();
        
        reader.onload = function(e) {
            // Fire-and-forget: clear server state
            API.fetchClear().catch(function() {});
            
            var imageData = e.target.result;
            
            // Store image
            State.setCurrentImage(imageData);
            
            // Update UI
            updateImagePreviews(imageData);
            
            // Reset state for new image
            Workflows.resetQueryState();
            
            // Enable buttons
            checkButtonsState();
            
            UI.showToast('Image uploaded successfully', 'success');
            
            // Show post-upload buttons
            showPostUploadButtons();
            
            // Process: Detect → Extract
            Workflows.processInputImage(imageData);
        };
        
        reader.onerror = function() {
            UI.showToast('Error reading file', 'error');
        };
        
        reader.readAsDataURL(file);
        event.target.value = '';
    }

    /**
     * Process input image through full pipeline
     * @param {string} imageData - Base64 image data
     * @returns {Promise<{success: boolean, hasFace: boolean, hasEmbedding: boolean}>}
     */
    function processInputImage(imageData) {
        // Step 1: Detect faces
        return Workflows.detectFaces()
            .then(function(detectResult) {
                if (!detectResult.success || !detectResult.hasFace) {
                    return { success: false, hasFace: false, hasEmbedding: false };
                }
                
                // Step 2: Extract features
                return Workflows.extractFeatures()
                    .then(function(extractResult) {
                        return {
                            success: extractResult.success,
                            hasFace: true,
                            hasEmbedding: extractResult.hasEmbedding
                        };
                    });
            });
    }

    // ============================================================================
    // 2. PUBLIC API - Detection
    // ============================================================================

    /**
     * Detect faces in current image
     * @returns {Promise<{success: boolean, hasFace: boolean, count?: number}>}
     */
    function detectFaces() {
        if (!State.hasQueryImage()) {
            UI.showToast('No image to detect', 'warning');
            return Promise.resolve({ success: false, hasFace: false });
        }
        
        UI.showLoading('Detecting faces...');
        
        return API.fetchDetect(State.getCurrentImage())
            .then(function(result) {
                UI.hideLoading();
                
                if (result.success && result.data && result.data.count > 0) {
                    // Store thumbnails - API returns { faces: [{ thumbnail, bbox, ... }] }
                    State.setFaceThumbnails(result.data.faces || []);
                    
                    // Update UI
                    updateFacesGallery(result.data.faces);
                    UI.updateStatus('detectStatus', 'Found ' + result.data.count + ' face(s)', 'success');
                    
                    return { success: true, hasFace: true, count: result.data.count };
                } else {
                    UI.updateStatus('detectStatus', 'No faces detected', 'warning');
                    UI.showToast('No faces detected', 'warning');
                    
                    return { success: true, hasFace: false };
                }
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.updateStatus('detectStatus', 'Error detecting faces', 'error');
                UI.showToast('Error: ' + err.message, 'error');
                
                return { success: false, hasFace: false };
            });
    }

    // ============================================================================
    // 3. PUBLIC API - Extraction
    // ============================================================================

    /**
     * Extract embeddings from detected face
     * @returns {Promise<{success: boolean, hasEmbedding: boolean}>}
     */
    function extractFeatures() {
        if (!State.hasDetectedFaces()) {
            UI.showToast('No face to extract', 'warning');
            return Promise.resolve({ success: false, hasEmbedding: false });
        }
        
        UI.showLoading('Extracting features...');
        
        return API.fetchExtract()
            .then(function(result) {
                UI.hideLoading();
                
                if (result.success && result.data) {
                    // Store embedding
                    State.setQueryEmbedding(result.data.embedding);
                    
                    // Cache visualizations if available
                    if (result.data.visualizations) {
                        for (var vizType in result.data.visualizations) {
                            if (result.data.visualizations.hasOwnProperty(vizType)) {
                                State.addVisualizationData(vizType, result.data.visualizations[vizType]);
                            }
                        }
                    }
                    
                    // Update UI
                    var dim = result.data.embedding_size || 128;
                    UI.updateStatus('extractStatus', 'Signature created (' + dim + '-dim)', 'success');
                    
                    return { success: true, hasEmbedding: true };
                } else {
                    UI.updateStatus('extractStatus', 'Failed to create signature', 'error');
                    UI.showToast(result.error || 'Extraction failed', 'error');
                    
                    return { success: false, hasEmbedding: false };
                }
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.updateStatus('extractStatus', 'Error creating signature', 'error');
                UI.showToast('Error: ' + err.message, 'error');
                
                return { success: false, hasEmbedding: false };
            });
    }

    // ============================================================================
    // 4. PUBLIC API - References
    // ============================================================================

    /**
     * Handle reference file selection
     * @param {Event} event - File input change event
     */
    function handleReferenceSelect(event) {
        var file = event.target.files[0];
        
        if (!file) return;
        
        var reader = new FileReader();
        
        reader.onload = function(e) {
            var imageData = e.target.result;
            
            // Prompt for name
            var name = prompt('Enter reference name:', 'Reference');
            
            if (!name) {
                UI.showToast('Name required', 'warning');
                return;
            }
            
            Workflows.saveReference(imageData, name);
        };
        
        reader.onerror = function() {
            UI.showToast('Error reading file', 'error');
        };
        
        reader.readAsDataURL(file);
        event.target.value = '';
    }

    /**
     * Save reference to backend
     * @param {string} imageData - Base64 image data
     * @param {string} name - Reference name
     * @returns {Promise<{success: boolean}>}
     */
    function saveReference(imageData, name) {
        name = name || 'Reference';
        
        UI.showLoading('Saving reference...');
        
        return API.fetchAddReference(imageData, name)
            .then(function(result) {
                UI.hideLoading();
                
                if (result.success && result.data) {
                    State.addReference(result.data.reference || result.data);
                    updateReferenceList();
                    UI.showToast('Reference saved', 'success');
                    
                    return { success: true };
                } else {
                    UI.showToast(result.error || 'Failed to save', 'error');
                    
                    return { success: false };
                }
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error: ' + err.message, 'error');
                
                return { success: false };
            });
    }

    /**
     * Remove reference
     * @param {number} index - Reference index
     */
    function removeReference(index) {
        UI.showLoading('Removing reference...');
        
        API.fetchRemoveReference(index)
            .then(function(result) {
                UI.hideLoading();
                
                if (result.success) {
                    State.removeReference(index);
                    updateReferenceList();
                    UI.showToast('Reference removed', 'success');
                } else {
                    UI.showToast(result.error || 'Failed to remove', 'error');
                }
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error: ' + err.message, 'error');
            });
    }

    // ============================================================================
    // 5. PUBLIC API - Library
    // ============================================================================

    /**
     * Load library persons
     * @returns {Promise}
     */
    function loadLibrary() {
        return API.fetchLibrary()
            .then(function(result) {
                if (result.success && result.data) {
                    State.setLibraryPersons(result.data.persons || []);
                    return State.getLibraryPersons();
                }
                return [];
            })
            .catch(function() {
                return [];
            });
    }

    /**
     * Handle library upload
     * @param {Event} event - File input change event
     */
    function handleLibraryUpload(event) {
        var file = event.target.files[0];
        
        if (!file) return;
        
        var reader = new FileReader();
        
        reader.onload = function(e) {
            State.setLibraryImageData(e.target.result);
            State.setLibrarySource('upload');
            // Trigger library modal
            showLibraryModal();
        };
        
        reader.readAsDataURL(file);
        event.target.value = '';
    }

    /**
     * Save to library
     * @param {string} name - Person name
     * @param {string} notes - Optional notes
     */
    function saveToLibrary(name, notes) {
        if (!name) {
            UI.showToast('Name required', 'warning');
            return;
        }
        
        var imageData = State.getLibraryImageData();
        if (!imageData) {
            UI.showToast('No image', 'warning');
            return;
        }
        
        UI.showLoading('Saving to library...');
        
        API.fetchAddPerson(name, imageData, notes || '')
            .then(function(result) {
                UI.hideLoading();
                
                if (result.success) {
                    UI.showToast('Saved to library', 'success');
                    closeLibraryModal();
                    return loadLibrary();
                } else {
                    UI.showToast(result.error || 'Failed to save', 'error');
                }
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error: ' + err.message, 'error');
            });
    }

    // ============================================================================
    // 6. PUBLIC API - Reset
    // ============================================================================

    /**
     * Clear all cached data
     */
    function clearAllCache() {
        UI.showLoading('Clearing cache...');
        
        API.fetchClear()
            .then(function() {
                // Reset state
                State.resetState();
                
                // Reset UI
                clearAllUIElements();
                
                UI.hideLoading();
                UI.showToast('Cache cleared', 'success');
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error clearing cache', 'error');
            });
    }

    /**
     * Reset query state only
     */
    function resetQueryState() {
        State.resetQueryState();
        
        // Reset UI elements
        var elements = ['facesContainer', 'detectStatus', 'extractStatus', 'compareStatus', 'step5Status'];
        
        elements.forEach(function(id) {
            var el = document.getElementById(id);
            
            if (el) {
                if (id.includes('Status')) {
                    el.textContent = 'Waiting...';
                    el.className = 'status';
                }
            }
        });
    }

    // ============================================================================
    // 7. INTERNAL - UI Updates
    // ============================================================================

    function updateImagePreviews(imageData) {
        var selectedImage = document.getElementById('selectedImage');
        var step5Image = document.getElementById('step5SelectedImage');
        
        if (selectedImage) selectedImage.src = imageData;
        if (step5Image) step5Image.src = imageData;
        
        var previewContainers = ['previewContainer', 'step5PreviewContainer'];
        
        previewContainers.forEach(function(id) {
            var container = document.getElementById(id);
            
            if (container) {
                container.classList.remove('hidden');
                container.classList.add('visible');
            }
        });
    }

    function updateFacesGallery(faces) {
        var gallery = document.getElementById('step5FacesGallery');
        var container = document.getElementById('step5FacesContainer');
        
        if (!gallery) return;
        
        gallery.innerHTML = '';
        
        if (faces) {
            faces.forEach(function(face, i) {
                var item = document.createElement('div');
                item.className = 'gallery-item';
                item.innerHTML = '<img src="data:image/png;base64,' + face.thumbnail + '" alt="Face ' + (i + 1) + '"><span>' + (i + 1) + '</span>';
                gallery.appendChild(item);
            });
        }
        
        if (container) {
            container.classList.remove('hidden');
            container.classList.add('visible');
        }
    }

    function updateReferenceList() {
        var refs = State.getReferences();
        var list = document.getElementById('referenceList');
        
        if (!list) return;
        
        if (refs.length === 0) {
            list.innerHTML = '<p class="empty-hint">No references yet</p>';
            return;
        }
        
        list.innerHTML = refs.map(function(ref, i) {
            var name = ref.name || 'Ref ' + (i + 1);
            var thumb = ref.thumbnail || '';
            return '<div class="reference-item" onclick="Workflows.showReferenceDetails(' + i + ', ' + JSON.stringify(ref).replace(/"/g, '&quot;') + ')">' +
                '<img src="data:image/png;base64,' + thumb + '" alt="' + name + '">' +
                '<span>' + name + '</span></div>';
        }).join('');
    }

    function checkButtonsState() {
        var hasImage = State.hasQueryImage();
        var hasFace = State.hasDetectedFaces();
        var hasEmbedding = State.hasQueryEmbedding();
        var hasRefs = State.hasReferences();
        var hasSelectedLibRefs = State.hasSelectedLibraryRefs();
        
        UI.setButtonEnabled('detectBtn', hasImage);
        UI.setButtonEnabled('extractBtn', hasFace);
        UI.setButtonEnabled('compareBtn', hasEmbedding && (hasRefs || hasSelectedLibRefs));
    }

    function showPostUploadButtons() {
        var initial = document.getElementById('step1ButtonsInitial');
        var after = document.getElementById('step1ButtonsAfter');
        
        if (initial) initial.classList.add('hidden');
        if (after) after.classList.remove('hidden');
    }

    function clearAllUIElements() {
        // Clear images
        var images = ['selectedImage', 'step5SelectedImage', 'queryImage', 'refImage'];
        images.forEach(function(id) { UI.clearImage(id); });
        
        // Hide containers
        var containers = ['previewContainer', 'step5PreviewContainer', 'facesContainer'];
        containers.forEach(function(id) { UI.hideElement(id); });
        
        // Clear text
        var texts = ['detectStatus', 'extractStatus', 'compareStatus', 'step5Status'];
        texts.forEach(function(id) { UI.setElementText(id, 'Waiting...'); });
        
        // Disable buttons
        var buttons = ['detectBtn', 'extractBtn', 'compareBtn'];
        buttons.forEach(function(id) { UI.setButtonEnabled(id, false); });
    }

    // ============================================================================
    // 8. EXTRAS - Library Modal
    // ============================================================================

    function showLibraryModal() {
        var modal = document.getElementById('libraryModal');
        if (modal) {
            modal.classList.add('active');
        }
    }

    function closeLibraryModal() {
        var modal = document.getElementById('libraryModal');
        if (modal) {
            modal.classList.remove('active');
        }
        State.setLibraryImageData(null);
    }

    // Placeholder functions - implement as needed
    function showReferenceDetails(index, ref) {
        console.log('Show reference details:', index, ref);
    }

    // ============================================================================
    // 9. EXPORTS
    // ============================================================================

    global.Workflows = {
        // Upload
        handleImageSelect: handleImageSelect,
        processInputImage: processInputImage,
        
        // Detection & Extraction
        detectFaces: detectFaces,
        extractFeatures: extractFeatures,
        
        // References
        handleReferenceSelect: handleReferenceSelect,
        saveReference: saveReference,
        removeReference: removeReference,
        
        // Library
        loadLibrary: loadLibrary,
        handleLibraryUpload: handleLibraryUpload,
        saveToLibrary: saveToLibrary,
        showLibraryModal: showLibraryModal,
        closeLibraryModal: closeLibraryModal,
        
        // Reset
        clearAllCache: clearAllCache,
        resetQueryState: resetQueryState,
        
        // Reference details
        showReferenceDetails: showReferenceDetails
    };

})(window);
