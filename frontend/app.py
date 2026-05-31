import os
import json
import base64
import math
import logging
import tempfile
import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from pathlib import Path
import glob
from collections import Counter
from scipy.spatial import ConvexHull
import torch
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
app.logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).parent.parent
PROJECT_DIR = BASE_DIR
RESULTS_DIR = PROJECT_DIR / "Results"
MODELS_DIR = RESULTS_DIR / "trained_models"
ML_RESULTS_DIR = RESULTS_DIR / "ml_results"
STATS_DIR = RESULTS_DIR / "statistical_analysis"
FRONTEND_DIR = Path(__file__).parent
SAMPLE_IMAGES_DIR = FRONTEND_DIR / "static" / "sample_images"

DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

models = {}
model_scalers = {}
feature_scaler = None
feature_selector = None
selected_features = []
raw_feature_names = []
yolo_model = None
sample_cache = {}

RAW_FEATURE_COLUMNS = [
    'vehicle_count', 'density_score', 'center_x_mean', 'center_y_mean',
    'bbox_area_mean', 'bbox_area_std', 'aspect_ratio_mean', 'spread_area',
    'spatial_entropy', 'cluster_score', 'road_utilization', 'count_car',
    'count_motorcycle', 'ratio_car', 'ratio_motorcycle', 'type_diversity',
    'heavy_vehicle_ratio', 'lane_occupancy', 'traffic_flow_score',
    'congestion_index', 'speed_estimate', 'confidence_mean', 'confidence_std'
]

print("=" * 60)
print("🚀 SKYTRAFFIC AI FLASK DASHBOARD STARTUP")
print("=" * 60)
print(f"Demo Mode: {DEMO_MODE}")

def load_models():
    global models, model_scalers, feature_scaler, feature_selector
    global selected_features, raw_feature_names, yolo_model, sample_cache

    print("📦 Loading ML models...")
    model_files = {
        'random_forest': 'random_forest_model.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl',
        'mlp': 'mlp_model.pkl',
        'svm': 'svm_model.pkl',
        'knn': 'knn_model.pkl',
        'linear_regression': 'linear_regression_model.pkl',
    }

    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  ✓ {name}")

    print("📊 Loading feature scaler and selector...")
    scaler_path = RESULTS_DIR / "feature_scaler.pkl"
    selector_path = MODELS_DIR / "feature_selector.pkl"
    selected_features_path = MODELS_DIR / "selected_features.pkl"

    if scaler_path.exists():
        feature_scaler = joblib.load(scaler_path)
        raw_feature_names = list(getattr(feature_scaler, 'feature_names_in_', RAW_FEATURE_COLUMNS))
        print(f"  ✓ feature_scaler")

    else:
        raw_feature_names = RAW_FEATURE_COLUMNS.copy()

    if selector_path.exists():
        feature_selector = joblib.load(selector_path)
        print(f"  ✓ feature_selector")

    print("📊 Loading model-specific scalers...")
    if selected_features_path.exists():
        selected_features = joblib.load(selected_features_path)
        print(f"  ✓ selected_features ({len(selected_features)})")

    scaler_names = ['logistic_regression_scaler', 'mlp_scaler', 'svm_scaler']
    for scaler_name in scaler_names:
        path = MODELS_DIR / f"{scaler_name}.pkl"
        if path.exists():
            model_scalers[scaler_name] = joblib.load(path)
            print(f"  ✓ {scaler_name}")

    print("🎯 Loading YOLO model...")
    yolo_weights = PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if yolo_weights.exists():
        yolo_model = YOLO(str(yolo_weights))
        print(f"  ✓ YOLO model loaded")

    print("💾 Loading demo cache...")
    cache_path = SAMPLE_IMAGES_DIR / "sample_cache.json"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            sample_cache = json.load(f)
        print(f"  ✓ Loaded {len(sample_cache)} cached samples")

load_models()

def calculate_spatial_entropy(centers_x, centers_y, img_width, img_height, grid_size=4):
    if not centers_x:
        return 0.0

    grid_counts = np.zeros((grid_size, grid_size))
    for x, y in zip(centers_x, centers_y):
        grid_x = min(int(x / img_width * grid_size), grid_size - 1)
        grid_y = min(int(y / img_height * grid_size), grid_size - 1)
        grid_counts[grid_y, grid_x] += 1

    total_points = len(centers_x)
    probabilities = grid_counts / total_points
    entropy = 0.0
    for p in probabilities.flatten():
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(grid_size * grid_size)
    return entropy / max_entropy if max_entropy > 0 else 0.0

def calculate_cluster_score(centers_x, centers_y):
    if len(centers_x) < 2:
        return 0.0

    centroid_x = np.mean(centers_x)
    centroid_y = np.mean(centers_y)
    distances = [math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
                for x, y in zip(centers_x, centers_y)]
    mean_distance = np.mean(distances)

    if centers_x:
        x_range = max(centers_x) - min(centers_x)
        y_range = max(centers_y) - min(centers_y)
        max_possible = math.sqrt(x_range**2 + y_range**2)
        if max_possible > 0:
            return 1 - (mean_distance / max_possible)
    return 0.0

def extract_spatial_features(valid_detections, img_shape):
    features = {}

    if not valid_detections:
        features.update({
            'vehicle_count': 0, 'density_score': 0.0, 'spread_area': 0.0,
            'center_x_mean': 0.5, 'center_y_mean': 0.5, 'bbox_area_mean': 0.0,
            'bbox_area_std': 0.0, 'aspect_ratio_mean': 0.0,
            'spatial_entropy': 0.0, 'cluster_score': 0.0, 'road_utilization': 0.0
        })
        return features

    img_height, img_width = img_shape[:2]
    vehicle_count = len(valid_detections)
    features['vehicle_count'] = vehicle_count
    features['density_score'] = vehicle_count / (img_width * img_height) * 100000

    centers_x, centers_y, bbox_areas, aspect_ratios = [], [], [], []

    for detection in valid_detections:
        bbox = detection['bbox']
        center_x, center_y, width, height = bbox
        center_x_px = center_x * img_width
        center_y_px = center_y * img_height
        width_px = width * img_width
        height_px = height * img_height

        centers_x.append(center_x_px)
        centers_y.append(center_y_px)
        bbox_areas.append(width_px * height_px)
        aspect_ratios.append(width_px / height_px if height_px > 0 else 0)

    if centers_x:
        features['center_x_mean'] = np.mean(centers_x) / img_width
        features['center_y_mean'] = np.mean(centers_y) / img_height
        features['bbox_area_mean'] = np.mean(bbox_areas)
        features['bbox_area_std'] = np.std(bbox_areas) if len(bbox_areas) > 1 else 0.0
        features['aspect_ratio_mean'] = np.mean(aspect_ratios)

        if len(centers_x) >= 3:
            try:
                points = np.column_stack([centers_x, centers_y])
                hull = ConvexHull(points)
                features['spread_area'] = hull.volume
            except:
                x_range = max(centers_x) - min(centers_x)
                y_range = max(centers_y) - min(centers_y)
                features['spread_area'] = x_range * y_range
        else:
            x_range = max(centers_x) - min(centers_x)
            y_range = max(centers_y) - min(centers_y)
            features['spread_area'] = x_range * y_range

        features['spatial_entropy'] = calculate_spatial_entropy(centers_x, centers_y, img_width, img_height)
        features['cluster_score'] = calculate_cluster_score(centers_x, centers_y)
        features['road_utilization'] = (max(centers_x) - min(centers_x)) / img_width if centers_x else 0.0
    else:
        features.update({
            'center_x_mean': 0.5, 'center_y_mean': 0.5, 'bbox_area_mean': 0.0,
            'bbox_area_std': 0.0, 'aspect_ratio_mean': 0.0, 'spread_area': 0.0,
            'spatial_entropy': 0.0, 'cluster_score': 0.0, 'road_utilization': 0.0
        })

    return features

def extract_vehicle_type_features(class_counts, actual_classes):
    features = {}
    total_vehicles = sum(class_counts.values())

    for class_name in actual_classes:
        features[f'count_{class_name}'] = class_counts.get(class_name, 0)

    if total_vehicles > 0:
        for class_name in actual_classes:
            features[f'ratio_{class_name}'] = class_counts.get(class_name, 0) / total_vehicles
    else:
        for class_name in actual_classes:
            features[f'ratio_{class_name}'] = 0.0

    type_entropy = 0.0
    if total_vehicles > 0:
        for count in class_counts.values():
            if count > 0:
                p = count / total_vehicles
                type_entropy -= p * math.log2(p)

    max_entropy = math.log2(len(actual_classes)) if len(actual_classes) > 0 else 1.0
    features['type_diversity'] = type_entropy / max_entropy if max_entropy > 0 else 0.0
    heavy_vehicles = class_counts.get('motorcycle', 0)
    features['heavy_vehicle_ratio'] = heavy_vehicles / total_vehicles if total_vehicles > 0 else 0.0

    return features

def extract_traffic_pattern_features(valid_detections, img_shape):
    features = {}

    if not valid_detections:
        features.update({
            'lane_occupancy': 0.0, 'traffic_flow_score': 0.0,
            'congestion_index': 0.0, 'speed_estimate': 50.0
        })
        return features

    img_height, img_width = img_shape[:2]
    vehicle_count = len(valid_detections)

    y_positions = [detection['bbox'][1] * img_height for detection in valid_detections]
    features['lane_occupancy'] = (max(y_positions) - min(y_positions)) / img_height if y_positions else 0.0

    x_positions = [detection['bbox'][0] * img_width for detection in valid_detections]
    x_variance = np.var(x_positions) / (img_width ** 2) if x_positions and img_width > 0 else 0.0
    features['traffic_flow_score'] = 1.0 - min(x_variance * 10, 1.0)

    density = vehicle_count / (img_width * img_height) * 100000
    spread = (max(x_positions) - min(x_positions)) / img_width if x_positions else 1.0
    features['congestion_index'] = density * (1 - spread)

    base_speed = 50
    if vehicle_count > 0:
        speed_factor = 1.0 - (vehicle_count / 50)
        features['speed_estimate'] = base_speed * speed_factor
    else:
        features['speed_estimate'] = base_speed

    return features

def extract_all_features(valid_detections, class_counts, img_shape, actual_classes):
    features = {}
    features['vehicle_count'] = len(valid_detections)

    spatial_features = extract_spatial_features(valid_detections, img_shape)
    features.update(spatial_features)

    type_features = extract_vehicle_type_features(class_counts, actual_classes)
    features.update(type_features)

    pattern_features = extract_traffic_pattern_features(valid_detections, img_shape)
    features.update(pattern_features)

    if valid_detections:
        confidences = [d['confidence'] for d in valid_detections]
        features['confidence_mean'] = np.mean(confidences)
        features['confidence_std'] = np.std(confidences) if len(confidences) > 1 else 0.0
    else:
        features['confidence_mean'] = 0.0
        features['confidence_std'] = 0.0

    return features

def run_yolo_detection(image_path):
    global yolo_model

    if yolo_model is None:
        return None, None, None, None

    results = yolo_model.predict(
        image_path,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        max_det=100,
        verbose=False
    )

    vehicle_count = 0
    class_counts = {'car': 0, 'motorcycle': 0}
    valid_detections = []

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywhn.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for bbox, cls, conf in zip(boxes, classes, confidences):
            cls_int = int(cls)
            class_name = 'car' if cls_int == 0 else 'motorcycle'

            if conf >= CONFIDENCE_THRESHOLD:
                vehicle_count += 1
                class_counts[class_name] += 1
                valid_detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': bbox
                })

    annotated_image = results[0].plot()
    return vehicle_count, class_counts, valid_detections, annotated_image

def classify_traffic_density(vehicle_count):
    LOW_THRESHOLD = 5
    MEDIUM_THRESHOLD = 20

    if vehicle_count <= LOW_THRESHOLD:
        return "Low"
    elif vehicle_count <= MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"

def get_predictions(features_dict):
    global models, model_scalers, feature_selector, feature_scaler, raw_feature_names

    predictions = {}

    expected_features = raw_feature_names or RAW_FEATURE_COLUMNS
    ordered_features = {
        col: float(features_dict.get(col, 0.0))
        for col in expected_features
    }
    raw_df = pd.DataFrame([ordered_features], columns=expected_features)

    if feature_scaler is not None:
        features_array = feature_scaler.transform(raw_df)
    else:
        features_array = raw_df.to_numpy(dtype=float)

    if feature_selector is not None:
        features_array = feature_selector.transform(features_array)

    for model_name, model in models.items():
        try:
            test_features = features_array.copy()

            if model_name in ['logistic_regression', 'mlp', 'svm']:
                scaler_key = f"{model_name}_scaler"
                if scaler_key in model_scalers:
                    test_features = model_scalers[scaler_key].transform(test_features)

            pred = model.predict(test_features)[0]
            if model_name == 'linear_regression':
                predictions['linear_regression'] = float(pred)
            else:
                predictions[model_name] = str(pred)
        except Exception as e:
            app.logger.exception(
                "Error predicting with %s: %s; features shape=%s",
                model_name,
                e,
                getattr(features_array, 'shape', None),
            )
            predictions[model_name] = "Error"

    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/models')
def models_page():
    return render_template('models.html')

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route('/stats')
def stats_page():
    return render_template('stats.html')

@app.route('/api/model-results')
def api_model_results():
    results_path = ML_RESULTS_DIR / "training_summary.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Results not found"}), 404

@app.route('/api/confusion-matrix/<model_name>')
def api_confusion_matrix(model_name):
    matrix_path = ML_RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    if matrix_path.exists():
        return send_file(matrix_path, mimetype='image/png')
    return jsonify({"error": "Confusion matrix not found"}), 404

@app.route('/api/stats-images')
def api_stats_images():
    images = []
    if STATS_DIR.exists():
        for img in sorted(STATS_DIR.glob("*.png")):
            images.append(img.name)
    return jsonify(images)

@app.route('/api/stats-image/<image_name>')
def api_stats_image(image_name):
    img_path = STATS_DIR / image_name
    if img_path.exists():
        return send_file(img_path, mimetype='image/png')
    return jsonify({"error": "Image not found"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    global yolo_model

    temp_path = None

    try:
        app.logger.info("Step 1: receiving file")

        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400

        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name

        if not cv2.imwrite(temp_path, img):
            raise RuntimeError("Could not write uploaded image to a temporary file")

        vehicle_count, class_counts, valid_detections, annotated_image = run_yolo_detection(temp_path)
        app.logger.info(f"Step 2: YOLO detected {vehicle_count} vehicles")

        if annotated_image is None:
            return jsonify({'status': 'error', 'message': 'YOLO detection failed'}), 500

        features = extract_all_features(valid_detections, class_counts, img.shape, ['car', 'motorcycle'])
        app.logger.info(f"Step 3: features extracted: {len(features)} features")
        predictions = get_predictions(features)
        app.logger.info(f"Step 4: predictions: {predictions}")
        traffic_label = classify_traffic_density(vehicle_count)

        _, img_buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        features_dict = {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v
                        for k, v in features.items()}

        lr_prediction = float(predictions.get('linear_regression', 0))
        lr_prediction = max(0, round(lr_prediction, 1))
        predictions['linear_regression'] = lr_prediction

        response = {
            'status': 'success',
            'image': f'data:image/jpeg;base64,{img_base64}',
            'vehicle_count': vehicle_count,
            'car_count': class_counts.get('car', 0),
            'motorcycle_count': class_counts.get('motorcycle', 0),
            'traffic_label': traffic_label,
            'predictions': predictions,
            'features': features_dict
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction failed at: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                app.logger.warning("Could not remove temporary image %s", temp_path)

@app.route('/demo/<image_name>')
def demo(image_name):
    global sample_cache

    if image_name not in sample_cache:
        return jsonify({"error": "Demo image not found"}), 404

    cached = sample_cache[image_name]
    img_path = SAMPLE_IMAGES_DIR / cached.get('image_path', f'{image_name}.jpg')

    if not img_path.exists():
        return jsonify({"error": "Image file not found"}), 404

    img = cv2.imread(str(img_path))
    _, img_buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_buffer).decode('utf-8')

    predictions = dict(cached.get('predictions', {}))
    if 'linear_regression' in cached:
        predictions['linear_regression'] = cached['linear_regression']

    response = {
        'status': 'success',
        'image': f'data:image/jpeg;base64,{img_base64}',
        'vehicle_count': int(cached.get('vehicle_count', 0)),
        'car_count': int(cached.get('car_count', 0)),
        'motorcycle_count': int(cached.get('motorcycle_count', cached.get('truck_count', 0))),
        'traffic_label': cached.get('traffic_label', 'Low'),
        'predictions': predictions,
        'features': dict(cached.get('features', {}))
    }
    return jsonify(response)

if __name__ == '__main__':
    print("🌐 Starting Flask server on http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host='localhost', port=5000)
