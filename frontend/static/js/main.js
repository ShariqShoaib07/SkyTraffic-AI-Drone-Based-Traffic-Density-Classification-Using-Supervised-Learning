document.addEventListener('DOMContentLoaded', function() {
    setupDemoToggle();
    setupUploadHandlers();
    setupSampleImages();
});

let demoMode = false;

function setupDemoToggle() {
    const toggle = document.getElementById('demoToggle');
    if (toggle) {
        toggle.addEventListener('change', function() {
            demoMode = this.checked;
            console.log('Demo mode:', demoMode);
        });
    }
}

function setupUploadHandlers() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    if (!uploadArea) return;

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) handleFileSelect(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        displayResults();
        showLoading();

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                displayPredictions(data);
                localStorage.setItem('lastAnalysis', JSON.stringify(data));
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
                hideLoading();
            }
        })
        .catch(err => {
            console.error('Error:', err);
            alert('Error uploading image: ' + err.message);
            hideLoading();
        });
    };
    reader.readAsDataURL(file);
}

function setupSampleImages() {
    const samples = document.querySelectorAll('.sample-thumb');
    samples.forEach(img => {
        img.addEventListener('click', function() {
            const sampleName = this.dataset.sample;
            if (demoMode) {
                loadDemoImage(sampleName);
            }
        });
    });
}

function loadDemoImage(sampleName) {
    displayResults();
    showLoading();

    fetch(`/demo/${sampleName}`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                displayPredictions(data);
                localStorage.setItem('lastAnalysis', JSON.stringify(data));
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
                hideLoading();
            }
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
        const models = ['random_forest', 'decision_tree', 'logistic_regression', 'mlp', 'svm', 'knn'];
        models.forEach(model => {
            const pred = data.predictions[model] || 'N/A';
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
        const lr = data.predictions.linear_regression;
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
