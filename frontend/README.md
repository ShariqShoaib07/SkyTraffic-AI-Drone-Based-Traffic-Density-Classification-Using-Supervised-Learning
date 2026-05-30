# SkyTraffic AI Flask Dashboard

Professional web dashboard for UAV-based traffic density classification.

## Features

- **Real-time Image Analysis** — Upload drone images for instant traffic prediction
- **YOLO Detection** — Vehicle detection with bounding box visualization
- **6 ML Classifiers** — Ensemble predictions (Random Forest, SVM, MLP, KNN, Decision Tree, Logistic Regression)
- **Feature Analysis** — 24 engineered traffic metrics per image
- **Model Comparison** — View accuracy, performance, and confusion matrices
- **Demo Mode** — Pre-computed results for fast presentations
- **Dark Professional UI** — Optimized for presentations and analysis

## Installation

```bash
cd frontend
pip install flask ultralytics opencv-python joblib scikit-learn pandas numpy torch torchvision scipy
```

## Running the Dashboard

### Standard Mode (Real Predictions)
```bash
cd frontend
python app.py
```

Then open **http://localhost:5000** in your browser.

### Demo Mode (Fast Presentation)
```bash
cd frontend
DEMO_MODE=true python app.py
```

In demo mode:
- Uses pre-cached predictions for instant results (<100ms)
- No YOLO inference running
- Perfect for live presentations
- Click sample images to load pre-computed results

## Pages

### 🏠 Home (`/`)
- Hero section with project overview
- Key statistics (15,070 images, 7 models, 146,848 detections)
- Pipeline visualization

### 📊 Analyze (`/analyze`)
- Drag-and-drop image upload
- 6 sample images for demo
- Vehicle detection with annotated image
- Traffic density classification (Low/Medium/High)
- All model predictions
- Feature values

### 🎯 Models (`/models`)
- Accuracy bar chart for all 6 classifiers
- Individual model cards with metrics
- Confusion matrices for each model
- Cross-validation scores

### ⚙️ Features (`/features`)
- Feature documentation (24 traffic metrics)
- Radar chart of last analyzed image
- Feature importance and descriptions

### 📈 Stats (`/stats`)
- Dataset overview (2,321 samples, 15 selected features)
- Traffic density distribution
- Statistical analysis plots

## API Endpoints

### Prediction
- **POST `/predict`** — Upload image, get predictions
- **GET `/demo/<image_name>`** — Get cached demo result

### Data
- **GET `/api/model-results`** — Training summary JSON
- **GET `/api/confusion-matrix/<model_name>`** — Confusion matrix PNG
- **GET `/api/stats-images`** — List of statistical plots
- **GET `/api/stats-image/<image_name>`** — Get specific stats image

## Architecture

```
frontend/
├── app.py                    — Flask backend with ML inference
├── templates/
│   ├── base.html            — Shared layout with navbar
│   ├── index.html           — Home page
│   ├── analyze.html         — Main prediction interface
│   ├── models.html          — Model comparison
│   ├── features.html        — Feature analysis
│   └── stats.html           — Statistics
├── static/
│   ├── css/style.css        — Dark theme styling
│   ├── js/main.js           — Image upload, Chart.js
│   └── sample_images/       — Sample images & cache
└── README.md
```

## Configuration

**Environment Variables:**
- `DEMO_MODE` — Set to `true` for demo mode (default: `false`)

**Feature Extraction:**
- 24 traffic metrics extracted from YOLO detections
- 15 features selected using SelectKBest
- Features scaled using StandardScaler
- Applied to all 6 classifiers

**Model Parameters:**
- YOLO Confidence Threshold: 0.25
- YOLO IOU Threshold: 0.45
- Image Size: 640x640

## Performance

- **Real Inference:** ~3-5 seconds per image (CPU), <1s (GPU)
- **Demo Mode:** <100ms (pre-cached)
- **Memory:** ~2GB for all models + YOLO

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Color Scheme

- Background: `#0f1117` (dark grey)
- Accent: `#00d4aa` (teal)
- Success: `#2ed573` (green)
- Warning: `#ffa502` (orange)
- Danger: `#ff4757` (red)

## Development

To modify the UI:
1. Edit HTML in `templates/`
2. Update CSS in `static/css/style.css`
3. Modify JS in `static/js/main.js`
4. No build process needed

## Troubleshooting

**Models not loading?**
```bash
python -c "import joblib; print(joblib.load('../Results/trained_models/random_forest_model.pkl'))"
```

**YOLO not found?**
```bash
python -c "from ultralytics import YOLO; print(YOLO('../runs/detect/train/weights/best.pt'))"
```

**Sample images not showing?**
- Ensure `.jpg` files exist in `static/sample_images/`
- Verify `sample_cache.json` is valid JSON

## License

Part of SkyTraffic AI — UAV Traffic Density Classification
