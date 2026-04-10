// ============================================================================
// FILE: 04-compare.js
// PURPOSE: All comparison logic - unified interface
// ============================================================================

(function(global) {
    'use strict';

    // ============================================================================
    // 1. CONFIGURATION
    // ============================================================================

    var SCORE_THRESHOLDS = {
        MATCH: 0.75,
        POSSIBLE: 0.55
    };

    // ============================================================================
    // 2. INTERNAL - Score Display
    // ============================================================================

    /**
     * Update score display element
     * @param {string} elementId - Element ID
     * @param {number|null} score - Score value
     */
    function updateScoreDisplay(elementId, score) {
        var el = document.getElementById(elementId);
        if (!el) return;

        if (score !== null && score !== undefined) {
            var percent = Math.round(score * 100);
            el.textContent = percent + '%';
        } else {
            el.textContent = 'N/A';
        }
    }

    /**
     * Get status class based on match status
     * @param {string} status - Match status
     * @returns {string}
     */
    function getStatusClass(status) {
        if (status === 'match') return 'match';
        if (status === 'possible') return 'possible';
        return 'no_match';
    }

    // ============================================================================
    // 3. PUBLIC API - Compare
    // ============================================================================

    /**
     * Clear comparison results
     */
    function clearComparisonResults() {
        // Reset score displays
        var scoreIds = [
            'arcfaceScore', 'facenetScore', 'normScore', 'multiPoseScore',
            'lbpScore', 'asymScore', 'activationScore', 'irisScore',
            'expressionScore', 'matchScore'
        ];

        scoreIds.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.textContent = '--%';
        });

        // Hide result container
        var resultEl = document.getElementById('comparisonResult');
        if (resultEl) {
            resultEl.classList.remove('visible', 'active');
        }

        // Clear images
        UI.clearImage('queryImage');
        UI.clearImage('refImage');
        UI.setElementText('refLabel', '');
        var statusEl = document.getElementById('matchStatus');
        if (statusEl) {
            statusEl.textContent = '--';
            statusEl.className = 'comparison-status';
        }

        // Collapse scores
        UI.collapseScoresDropdown();
    }

    /**
     * Compare query with selected references (library or temporary)
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function compareFaces() {
        clearComparisonResults();
        UI.showLoading('Comparing...');

        // Validation
        if (!State.hasQueryEmbedding()) {
            UI.hideLoading();
            UI.showToast('Extract features first!', 'error');
            return Promise.resolve({ success: false, error: 'No embedding extracted' });
        }

        var hasLibraryRefs = State.hasSelectedLibraryRefs();
        var hasTempRefs = State.hasReferences();

        if (!hasLibraryRefs && !hasTempRefs) {
            UI.hideLoading();
            UI.showToast('Select a reference to compare', 'warning');
            return Promise.resolve({ success: false, error: 'No reference selected' });
        }

        // Choose comparison method
        if (hasLibraryRefs) {
            return compareWithSelectedLibraryRefs();
        } else {
            return compareWithTemporaryReferences();
        }
    }

    /**
     * Compare query with entire library
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function compareWithLibrary() {
        if (!State.hasQueryImage()) {
            UI.showToast('No image to compare', 'warning');
            return Promise.resolve({ success: false, error: 'No image' });
        }

        clearComparisonResults();
        UI.showLoading('Comparing with library...');

        return API.fetchLibraryMatch(State.getCurrentImage())
            .then(function(result) {
                UI.hideLoading();

                if (result.success && result.data && result.data.matches && result.data.matches.length > 0) {
                    var best = result.data.matches[0];
                    displayLibraryMatchResult(best);
                    return { success: true, data: best };
                } else {
                    UI.showToast('No matches found', 'info');
                    return { success: false, error: 'No matches' };
                }
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error: ' + err.message, 'error');
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 4. INTERNAL - Compare Helpers
    // ============================================================================

    /**
     * Compare with selected library persons
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function compareWithSelectedLibraryRefs() {
        var personIds = State.getSelectedLibraryRefs();
        var personNames = State.getSelectedLibraryRefNames();

        var promises = personIds.map(function(personId, index) {
            return API.fetchCompareLibrary(personId).then(function(result) {
                return {
                    name: personNames[index],
                    personId: personId,
                    result: result
                };
            });
        });

        return Promise.all(promises)
            .then(function(results) {
                var bestOverall = null;
                var bestScore = -1;

                results.forEach(function(item) {
                    if (item.result.success && item.result.data && item.result.data.best_match) {
                        var match = item.result.data.best_match;
                        var score = match.final_score || match.score || 0;

                        if (score > bestScore) {
                            bestScore = score;
                            bestOverall = match;
                        }
                    }
                });

                UI.hideLoading();

                if (bestOverall) {
                    displayComparisonResult(bestOverall);
                    return { success: true, data: bestOverall };
                }

                UI.showToast('No matches found', 'info');
                return { success: false, error: 'No matches found' };
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error: ' + err.message, 'error');
                return { success: false, error: err.message };
            });
    }

    /**
     * Compare with temporary references
     * @returns {Promise<{success: boolean, data?: object, error?: string}>}
     */
    function compareWithTemporaryReferences() {
        return API.fetchCompare()
            .then(function(result) {
                UI.hideLoading();

                if (result.success && result.data && result.data.best_match) {
                    displayComparisonResult(result.data.best_match);
                    return { success: true, data: result.data.best_match };
                }

                UI.showToast('No matches found', 'info');
                return { success: false, error: 'No matches found' };
            })
            .catch(function(err) {
                UI.hideLoading();
                UI.showToast('Error: ' + err.message, 'error');
                return { success: false, error: err.message };
            });
    }

    // ============================================================================
    // 5. INTERNAL - Display
    // ============================================================================

    /**
     * Display comparison result
     * @param {object} match - Match result
     */
    function displayComparisonResult(match) {
        // Update status
        UI.setElementText('matchStatus', match.match_label || 'No Match');
        var statusEl = document.getElementById('matchStatus');
        if (statusEl) {
            statusEl.className = 'comparison-status ' + getStatusClass(match.status);
        }

        // Update scores - handle different API field names
        updateScoreDisplay('arcfaceScore', match.arcface_similarity);
        updateScoreDisplay('facenetScore', match.facenet_similarity);
        updateScoreDisplay('activationScore', match.activation_similarity);
        updateScoreDisplay('irisScore', match.iris_similarity);
        updateScoreDisplay('expressionScore', match.expression_similarity);
        updateScoreDisplay('normScore', match.normalized_similarity);
        updateScoreDisplay('multiPoseScore', match.multi_pose_score);
        updateScoreDisplay('lbpScore', match.lbp_similarity || match.texture_similarity);
        updateScoreDisplay('asymScore', match.asymmetry_similarity || match.uniqueness_similarity);
        updateScoreDisplay('matchScore', match.final_score);

        // Show images - handle different API field names
        var thumbnails = State.getFaceThumbnails();
        if (thumbnails.length > 0 && thumbnails[0].thumbnail) {
            UI.setImageSrc('queryImage', 'data:image/png;base64,' + thumbnails[0].thumbnail);
        } else if (State.getCurrentImage()) {
            UI.setImageSrc('queryImage', State.getCurrentImage());
        }

        if (match.thumbnail) {
            UI.setImageSrc('refImage', 'data:image/png;base64,' + match.thumbnail);
        }

        UI.setElementText('refLabel', match.name || 'Reference');

        // Show result container
        var resultEl = document.getElementById('comparisonResult');
        if (resultEl) {
            resultEl.classList.add('visible', 'active');
            resultEl.style.display = 'flex';
        }

        // Expand scores
        UI.expandScoresDropdown();

        // Show toast
        var scorePercent = Math.round((match.final_score || 0) * 100);
        UI.showToast(match.name + ': ' + scorePercent + '%', 'success');
    }

    /**
     * Display library match result
     * @param {object} match - Match result
     */
    function displayLibraryMatchResult(match) {
        // Handle different field names from library/match endpoint
        var personName = match.person_name || match.name || 'Unknown';
        var score = match.score || match.final_score || 0;
        var scorePercent = Math.round(score * 100);

        // Update status
        var statusEl = document.getElementById('matchStatus');

        if (scorePercent >= 60) {
            UI.setElementText('matchStatus', personName);
            UI.showToast('Best match: ' + personName, 'success');
        } else {
            UI.setElementText('matchStatus', 'No Match');
            UI.showToast('No match found (below 60%)', 'info');
        }

        if (statusEl) {
            statusEl.className = 'comparison-status ' + (scorePercent >= 60 ? 'match' : 'no_match');
        }

        UI.setElementText('matchScore', scorePercent + '%');

        // Show images
        if (State.getCurrentImage()) {
            UI.setImageSrc('queryImage', State.getCurrentImage());
        }

        // Handle best_image from library/match
        if (match.best_image && match.best_image.thumbnail) {
            UI.setImageSrc('refImage', 'data:image/jpeg;base64,' + match.best_image.thumbnail);
        } else if (match.thumbnail) {
            UI.setImageSrc('refImage', 'data:image/png;base64,' + match.thumbnail);
        }

        UI.setElementText('refLabel', personName);

        // Show result container
        var resultEl = document.getElementById('comparisonResult');
        if (resultEl) {
            resultEl.classList.add('visible', 'active');
            resultEl.style.display = 'flex';
        }

        // Expand scores
        UI.expandScoresDropdown();
    }

    // ============================================================================
    // 6. EXPORTS
    // ============================================================================

    global.Compare = {
        clearComparisonResults: clearComparisonResults,
        compareFaces: compareFaces,
        compareWithLibrary: compareWithLibrary
    };

})(window);
