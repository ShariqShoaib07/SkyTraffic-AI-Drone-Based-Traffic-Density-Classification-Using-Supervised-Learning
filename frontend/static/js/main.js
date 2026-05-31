document.addEventListener('DOMContentLoaded', function() {
    setupDemoToggle();
    setupUploadHandlers();
    setupSampleImages();
});

let demoMode = false;
let selectedFile = null;

function setupDemoToggle() {
    const toggle = document.getElementById('demoToggle');
    if (toggle) {
        demoMode = toggle.checked;
        toggle.addEventListener('change', function() {
            demoMode = this.checked;
            console.log('Demo mode:', demoMode);
        });
    }
}

function setupUploadHandlers() {
    const chooseButton = document.getElementById('chooseImageBtn');
    const fileInput = document.getElementById('fileInput');
    const analyzeButton = document.getElementById('analyzeBtn');

    if (!chooseButton || !fileInput || !analyzeButton) return;

    chooseButton.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            previewSelectedFile(e.target.files[0]);
        }
    });

    analyzeButton.addEventListener('click', analyzeSelectedFile);
}

function previewSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    selectedFile = file;

    const fileName = document.getElementById('selectedFileName');
    const previewImage = document.getElementById('imagePreview');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const analyzeButton = document.getElementById('analyzeBtn');

    if (fileName) fileName.textContent = file.name;
    if (analyzeButton) analyzeButton.disabled = false;

    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImage) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        }
        if (previewPlaceholder) {
            previewPlaceholder.style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
}

function analyzeSelectedFile() {
    if (!selectedFile) {
        alert('Please choose an image first');
        return;
    }

    displayResults();
    showLoading();

    const formData = new FormData();
    formData.append('file', selectedFile);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(async (res) => {
        const data = await res.json();
        if (!res.ok || data.status !== 'success') {
            throw new Error(data.message || data.error || 'Unknown error');
        }
        return data;
    })
    .then(data => {
        displayPredictions(data);
        localStorage.setItem('lastAnalysis', JSON.stringify(data));
    })
    .catch(err => {
        console.error('Error:', err);
        alert('Error analyzing image: ' + err.message);
        hideLoading();
    });
}

function setupSampleImages() {
    const samples = document.querySelectorAll('.sample-thumb');
    samples.forEach(img => {
        img.addEventListener('click', function() {
            const sampleName = this.dataset.sample;
            showSamplePreview(sampleName, this.src);
            loadDemoImage(sampleName);
        });
    });
}

function showSamplePreview(sampleName, src) {
    selectedFile = null;

    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('selectedFileName');
    const previewImage = document.getElementById('imagePreview');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const analyzeButton = document.getElementById('analyzeBtn');

    if (fileInput) fileInput.value = '';
    if (fileName) fileName.textContent = sampleName;
    if (analyzeButton) analyzeButton.disabled = true;
    if (previewImage) {
        previewImage.src = src;
        previewImage.style.display = 'block';
    }
    if (previewPlaceholder) {
        previewPlaceholder.style.display = 'none';
    }
}

function loadDemoImage(sampleName) {
    displayResults();
    showLoading();

    fetch(`/demo/${sampleName}`)
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok || data.status !== 'success') {
                throw new Error(data.message || data.error || 'Unknown error');
            }
            return data;
        })
        .then(data => {
            displayPredictions(data);
            localStorage.setItem('lastAnalysis', JSON.stringify(data));
        })
        .catch(err => {
            console.error('Error:', err);
            alert('Error loading demo image: ' + err.message);
            hideLoading();
        });
}

function displayResults() {
    const panel = document.getElementById('resultsPanel');
    if (panel) {
        panel.style.display = 'block';
    }
}

function showLoading() {
    const loading = document.getElementById('loadingSpinner');
    const content = document.getElementById('resultsContent');
    if (loading) loading.style.display = 'flex';
    if (content) content.style.display = 'none';
}

function hideLoading() {
    const loading = document.getElementById('loadingSpinner');
    const content = document.getElementById('resultsContent');
    if (loading) loading.style.display = 'none';
    if (content) content.style.display = 'block';
}

function displayPredictions(data) {
    const image = document.getElementById('annotatedImage');
    if (image) {
        image.src = data.image;
    }

    const badge = document.getElementById('trafficBadge');
    const density = document.getElementById('densityValue');
    if (badge && density) {
        badge.classList.remove('medium', 'high');
        density.textContent = data.traffic_label;
        if (data.traffic_label === 'Medium') {
            badge.classList.add('medium');
        } else if (data.traffic_label === 'High') {
            badge.classList.add('high');
        }
    }

    document.getElementById('totalCount').textContent = data.vehicle_count;
    document.getElementById('carCount').textContent = data.car_count;
    document.getElementById('truckCount').textContent = data.truck_count;

    const grid = document.getElementById('predictionsGrid');
    if (grid) {
        grid.innerHTML = '';
        const modelNames = ['random_forest', 'decision_tree', 'logistic_regression', 'mlp', 'svm', 'knn'];
        modelNames.forEach(model => {
            const pred = data.predictions && data.predictions[model] ? data.predictions[model] : 'N/A';
            const badge = document.createElement('div');
            badge.className = 'prediction-badge';
            badge.innerHTML = `
                <div class="prediction-model">${model.replace(/_/g, ' ').toUpperCase()}</div>
                <div class="prediction-value">${pred}</div>
            `;
            grid.appendChild(badge);
        });
    }

    const lrCount = document.getElementById('lrCount');
    if (lrCount) {
        const lr = data.predictions ? data.predictions.linear_regression : null;
        lrCount.textContent = typeof lr === 'number' ? lr.toFixed(1) : 'N/A';
    }

    hideLoading();
}

window.addEventListener('pageshow', function() {
    const demoToggle = document.getElementById('demoToggle');
    if (demoToggle) {
        demoMode = demoToggle.checked;
    }
});
