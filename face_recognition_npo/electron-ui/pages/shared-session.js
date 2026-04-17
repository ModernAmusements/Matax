const API_BASE = 'http://localhost:3000/api';

class MantaxSessionState {
    static PREFIX = 'mantax_';
    static STORAGE = sessionStorage;

    static KEYS = {
        QUERY_IMAGE: 'query_image',
        QUERY_EMBEDDING: 'query_embedding',
        DETECTED_FACES: 'detected_faces',
        SELECTED_REFERENCES: 'selected_references',
        COMPARISON_RESULTS: 'comparison_results',
        TEMP_REFS: 'temp_refs'
    };

    static set(key, value) {
        try {
            const fullKey = this.PREFIX + key;
            const serialized = JSON.stringify(value);
            this.STORAGE.setItem(fullKey, serialized);
            return true;
        } catch (e) {
            console.error('MantaxSessionState.set error:', e);
            return false;
        }
    }

    static get(key, defaultValue = null) {
        try {
            const fullKey = this.PREFIX + key;
            const stored = this.STORAGE.getItem(fullKey);
            if (stored === null) return defaultValue;
            return JSON.parse(stored);
        } catch (e) {
            console.error('MantaxSessionState.get error:', e);
            return defaultValue;
        }
    }

    static remove(key) {
        try {
            const fullKey = this.PREFIX + key;
            this.STORAGE.removeItem(fullKey);
            return true;
        } catch (e) {
            console.error('MantaxSessionState.remove error:', e);
            return false;
        }
    }

    static clearQuery() {
        this.remove(this.KEYS.QUERY_IMAGE);
        this.remove(this.KEYS.QUERY_EMBEDDING);
        this.remove(this.KEYS.DETECTED_FACES);
    }

    static clearComparison() {
        this.remove(this.KEYS.COMPARISON_RESULTS);
        this.remove(this.KEYS.SELECTED_REFERENCES);
    }

    static clearAll() {
        Object.values(this.KEYS).forEach(key => this.remove(key));
    }

    static clearPageData() {
        this.clearQuery();
        this.clearComparison();
        this.remove(this.KEYS.TEMP_REFS);
    }

    static getQueryState() {
        return {
            image: this.get(this.KEYS.QUERY_IMAGE),
            embedding: this.get(this.KEYS.QUERY_EMBEDDING),
            faces: this.get(this.KEYS.DETECTED_FACES)
        };
    }

    static setQueryState(image, embedding, faces) {
        this.set(this.KEYS.QUERY_IMAGE, image);
        this.set(this.KEYS.QUERY_EMBEDDING, embedding);
        this.set(this.KEYS.DETECTED_FACES, faces);
    }

    static getComparisonState() {
        return {
            results: this.get(this.KEYS.COMPARISON_RESULTS),
            references: this.get(this.KEYS.SELECTED_REFERENCES)
        };
    }

    static setComparisonState(results, references) {
        this.set(this.KEYS.COMPARISON_RESULTS, results);
        this.set(this.KEYS.SELECTED_REFERENCES, references);
    }

    static isStorageAvailable() {
        try {
            const test = '__storage_test__';
            this.STORAGE.setItem(test, test);
            this.STORAGE.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }

    static getStorageInfo() {
        let used = 0;
        for (const key in this.STORAGE) {
            if (this.STORAGE.hasOwnProperty(key)) {
                used += this.STORAGE[key].length + key.length;
            }
        }
        return {
            used: (used / 1024).toFixed(2) + ' KB',
            available: '~5MB limit'
        };
    }
}

async function mantaxFetch(endpoint, options = {}) {
    const url = API_BASE + endpoint;
    const defaultOptions = {
        headers: { 'Content-Type': 'application/json' }
    };
    const finalOptions = { ...defaultOptions, ...options };

    try {
        const response = await fetch(url, finalOptions);
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'API request failed');
        return data;
    } catch (e) {
        console.error('mantaxFetch error:', e);
        throw e;
    }
}

function extractBase64(dataUrl) {
    if (!dataUrl) return null;
    if (dataUrl.includes(',')) return dataUrl.split(',')[1];
    return dataUrl;
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) { alert(message); return; }
    toast.textContent = message;
    toast.className = `toast toast-${type}`;
    toast.style.display = 'block';
    setTimeout(() => toast.style.display = 'none', 3000);
}

function formatPercentage(value) {
    if (value === null || value === undefined) return 'N/A';
    return (value * 100).toFixed(1) + '%';
}

function getConfidenceBand(score) {
    if (score >= 0.70) return { label: 'Very High', color: '#34C759' };
    if (score >= 0.45) return { label: 'High', color: '#30D158' };
    if (score >= 0.30) return { label: 'Moderate', color: '#FFD60A' };
    if (score >= 0.20) return { label: 'Low', color: '#FF9F0A' };
    return { label: 'Insufficient', color: '#FF3B30' };
}

function navigateTo(page) {
    window.location.href = page;
}