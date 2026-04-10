// ============================================================================
// FILE: 02-state.js
// PURPOSE: Centralized application state container
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. STATE STORE
    // ============================================================================

    var state = {
        // Query image state
        currentImage: null,
        currentFaceThumbnails: [],
        currentQueryEmbedding: null,

        // Temporary references
        references: [],
        selectedReferenceId: null,

        // Visualization cache
        visualizationData: {},

        // Library state
        libraryPersons: [],
        selectedLibraryRefs: [],
        selectedLibraryRefNames: [],

        // Webcam state
        webcamStream: null,
        faceMesh: null,
        meshOverlayActive: false,
        profileCaptures: [],
        autoCapturedFrames: [],
        autoCaptureInProgress: false,

        // Library modal state
        libraryImageData: null,
        librarySource: 'upload',

        // UI state
        terminalExpanded: false,
        sidebarOpen: false,
        currentWebcamActive: false,

        // Face mesh
        meshCamera: null
    };

    // ============================================================================
    // 2. GETTERS
    // ============================================================================

    function getState() {
        return state;
    }

    function getCurrentImage() {
        return state.currentImage;
    }

    function getFaceThumbnails() {
        return state.currentFaceThumbnails.slice();
    }

    function getQueryEmbedding() {
        return state.currentQueryEmbedding;
    }

    function getReferences() {
        return state.references.slice();
    }

    function getSelectedReferenceId() {
        return state.selectedReferenceId;
    }

    function getVisualizationData() {
        return state.visualizationData;
    }

    function getVisualization(vizType) {
        return state.visualizationData[vizType] || null;
    }

    function getLibraryPersons() {
        return state.libraryPersons.slice();
    }

    function getSelectedLibraryRefs() {
        return state.selectedLibraryRefs.slice();
    }

    function getSelectedLibraryRefNames() {
        return state.selectedLibraryRefNames.slice();
    }

    function isLibraryRefSelected(personId) {
        return state.selectedLibraryRefs.indexOf(personId) !== -1;
    }

    function getWebcamStream() {
        return state.webcamStream;
    }

    function isMeshOverlayActive() {
        return state.meshOverlayActive;
    }

    function getFaceMesh() {
        return state.faceMesh;
    }

    function getMeshCamera() {
        return state.meshCamera;
    }

    function getAutoCapturedFrames() {
        return state.autoCapturedFrames.slice();
    }

    function isAutoCaptureInProgress() {
        return state.autoCaptureInProgress;
    }

    function getLibraryImageData() {
        return state.libraryImageData;
    }

    function getLibrarySource() {
        return state.librarySource;
    }

    function isTerminalExpanded() {
        return state.terminalExpanded;
    }

    function isSidebarOpen() {
        return state.sidebarOpen;
    }

    function hasQueryImage() {
        return state.currentImage !== null;
    }

    function hasDetectedFaces() {
        return state.currentFaceThumbnails.length > 0;
    }

    function hasQueryEmbedding() {
        return state.currentQueryEmbedding !== null;
    }

    function hasReferences() {
        return state.references.length > 0;
    }

    function hasSelectedLibraryRefs() {
        return state.selectedLibraryRefs.length > 0;
    }

    function hasLibraryPersons() {
        return state.libraryPersons.length > 0;
    }

    // ============================================================================
    // 3. SETTERS
    // ============================================================================

    function setCurrentImage(imageData) {
        state.currentImage = imageData;
    }

    function setFaceThumbnails(thumbnails) {
        state.currentFaceThumbnails = thumbnails || [];
    }

    function setQueryEmbedding(embedding) {
        state.currentQueryEmbedding = embedding;
    }

    function setVisualizationData(vizType, data) {
        state.visualizationData[vizType] = data;
    }

    function setLibraryPersons(persons) {
        state.libraryPersons = persons || [];
    }

    function setLibraryImageData(imageData) {
        state.libraryImageData = imageData;
    }

    function setLibrarySource(source) {
        state.librarySource = source;
    }

    function setWebcamStream(stream) {
        state.webcamStream = stream;
    }

    function setMeshOverlayActive(active) {
        state.meshOverlayActive = active;
    }

    function setFaceMesh(mesh) {
        state.faceMesh = mesh;
    }

    function setMeshCamera(camera) {
        state.meshCamera = camera;
    }

    function setAutoCaptureInProgress(inProgress) {
        state.autoCaptureInProgress = inProgress;
    }

    function setTerminalExpanded(expanded) {
        state.terminalExpanded = expanded;
    }

    function setSidebarOpen(open) {
        state.sidebarOpen = open;
    }

    function setCurrentWebcamActive(active) {
        state.currentWebcamActive = active;
    }

    // ============================================================================
    // 4. STATE MUTATIONS
    // ============================================================================

    function addReference(ref) {
        state.references.push(ref);
    }

    function removeReference(index) {
        if (index >= 0 && index < state.references.length) {
            state.references.splice(index, 1);
        }
    }

    function clearReferences() {
        state.references = [];
    }

    function setSelectedReferenceId(id) {
        state.selectedReferenceId = id;
    }

    function toggleLibraryRef(personId, personName) {
        var index = state.selectedLibraryRefs.indexOf(personId);

        if (index > -1) {
            state.selectedLibraryRefs.splice(index, 1);
            state.selectedLibraryRefNames.splice(index, 1);
        } else {
            state.selectedLibraryRefs.push(personId);
            state.selectedLibraryRefNames.push(personName);
        }
    }

    function clearLibrarySelection() {
        state.selectedLibraryRefs = [];
        state.selectedLibraryRefNames = [];
    }

    function addAutoCapturedFrame(frame) {
        state.autoCapturedFrames.push(frame);
    }

    function clearAutoCapturedFrames() {
        state.autoCapturedFrames = [];
    }

    function addProfileCapture(capture) {
        state.profileCaptures.push(capture);
    }

    function clearProfileCaptures() {
        state.profileCaptures = [];
    }

    function addVisualizationData(vizType, data) {
        state.visualizationData[vizType] = data;
    }

    function clearVisualizationData() {
        state.visualizationData = {};
    }

    function addLibraryPerson(person) {
        state.libraryPersons.push(person);
    }

    function removeLibraryPerson(personId) {
        var index = -1;
        for (var i = 0; i < state.libraryPersons.length; i++) {
            if (state.libraryPersons[i].id === personId) {
                index = i;
                break;
            }
        }
        if (index !== -1) {
            state.libraryPersons.splice(index, 1);
        }
    }

    // ============================================================================
    // 5. RESET
    // ============================================================================

    /**
     * Reset all state to initial values
     */
    function resetState() {
        // Stop webcam if running
        if (state.webcamStream) {
            state.webcamStream.getTracks().forEach(function(track) {
                track.stop();
            });
            state.webcamStream = null;
        }

        // Reset face mesh
        if (state.meshCamera) {
            state.meshCamera.stop();
            state.meshCamera = null;
        }
        state.faceMesh = null;
        state.meshOverlayActive = false;

        // Reset all state
        state.currentImage = null;
        state.currentFaceThumbnails = [];
        state.currentQueryEmbedding = null;
        state.references = [];
        state.selectedReferenceId = null;
        state.visualizationData = {};
        state.selectedLibraryRefs = [];
        state.selectedLibraryRefNames = [];
        state.autoCapturedFrames = [];
        state.autoCaptureInProgress = false;
        state.libraryImageData = null;
        state.librarySource = 'upload';
    }

    /**
     * Reset only query state (keep library)
     */
    function resetQueryState() {
        state.currentImage = null;
        state.currentFaceThumbnails = [];
        state.currentQueryEmbedding = null;
        state.references = [];
        state.selectedReferenceId = null;
        state.visualizationData = {};
    }

    // ============================================================================
    // 6. EXPORTS
    // ============================================================================

    global.State = {
        // Getters
        getState: getState,
        getCurrentImage: getCurrentImage,
        getFaceThumbnails: getFaceThumbnails,
        getQueryEmbedding: getQueryEmbedding,
        getReferences: getReferences,
        getSelectedReferenceId: getSelectedReferenceId,
        getVisualizationData: getVisualizationData,
        getVisualization: getVisualization,
        getLibraryPersons: getLibraryPersons,
        getSelectedLibraryRefs: getSelectedLibraryRefs,
        getSelectedLibraryRefNames: getSelectedLibraryRefNames,
        isLibraryRefSelected: isLibraryRefSelected,
        getWebcamStream: getWebcamStream,
        isMeshOverlayActive: isMeshOverlayActive,
        getFaceMesh: getFaceMesh,
        getMeshCamera: getMeshCamera,
        getAutoCapturedFrames: getAutoCapturedFrames,
        isAutoCaptureInProgress: isAutoCaptureInProgress,
        getLibraryImageData: getLibraryImageData,
        getLibrarySource: getLibrarySource,
        isTerminalExpanded: isTerminalExpanded,
        isSidebarOpen: isSidebarOpen,
        hasQueryImage: hasQueryImage,
        hasDetectedFaces: hasDetectedFaces,
        hasQueryEmbedding: hasQueryEmbedding,
        hasReferences: hasReferences,
        hasSelectedLibraryRefs: hasSelectedLibraryRefs,
        hasLibraryPersons: hasLibraryPersons,

        // Setters
        setCurrentImage: setCurrentImage,
        setFaceThumbnails: setFaceThumbnails,
        setQueryEmbedding: setQueryEmbedding,
        setVisualizationData: setVisualizationData,
        setLibraryPersons: setLibraryPersons,
        setLibraryImageData: setLibraryImageData,
        setLibrarySource: setLibrarySource,
        setWebcamStream: setWebcamStream,
        setMeshOverlayActive: setMeshOverlayActive,
        setFaceMesh: setFaceMesh,
        setMeshCamera: setMeshCamera,
        setAutoCaptureInProgress: setAutoCaptureInProgress,
        setTerminalExpanded: setTerminalExpanded,
        setSidebarOpen: setSidebarOpen,
        setCurrentWebcamActive: setCurrentWebcamActive,

        // Mutations
        addReference: addReference,
        removeReference: removeReference,
        clearReferences: clearReferences,
        setSelectedReferenceId: setSelectedReferenceId,
        toggleLibraryRef: toggleLibraryRef,
        clearLibrarySelection: clearLibrarySelection,
        addAutoCapturedFrame: addAutoCapturedFrame,
        clearAutoCapturedFrames: clearAutoCapturedFrames,
        addProfileCapture: addProfileCapture,
        clearProfileCaptures: clearProfileCaptures,
        addVisualizationData: addVisualizationData,
        clearVisualizationData: clearVisualizationData,
        addLibraryPerson: addLibraryPerson,
        removeLibraryPerson: removeLibraryPerson,

        // Reset
        resetState: resetState,
        resetQueryState: resetQueryState
    };

})(window);
