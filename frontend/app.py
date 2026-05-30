import os
import json
import base64
import math
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

BASE_DIR = Path(__file__).parent.parent
PROJECT_DIR = BASE_DIR
RESULTS_DIR = PROJECT_DIR / "Results"
MODELS_DIR = RESULTS_DIR / "trained_models"
ML_RESULTS_DIR = RESULTS_DIR / "ml_results"
STATS_DIR = RESULTS_DIR / "statistical_analysis"
FRONTEND_DIR = Path(__file__).parent
SAMPLE_IMAGES_DIR = FRONTEND_DIR / "static" / "sample_images"

DEMO_MODE = os.getenv('DEMO_MODE', 'true').lower() == 'true'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

models = {}
model_scalers = {}
feature_scaler = None
feature_selector = None
yolo_model = None
sample_cache = {}

print("=" * 60)
print("🚀 SKYTRAFFIC AI FLASK DASHBOARD STARTUP")
print("=" * 60)
print(f"Demo Mode: {DEMO_MODE}")

def load_models():
    global models, model_scalers, feature_scaler, feature_selector, yolo_model, sample_cache

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
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    selector_path = MODELS_DIR / "feature_selector.pkl"

    if scaler_path.exists():
        feature_scaler = joblib.load(scaler_path)
        print(f"  ✓ feature_scaler")

    if selector_path.exists():
        feature_selector = joblib.load(selector_path)
        print(f"  ✓ feature_selector")

    print("📊 Loading model-specific scalers...")
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
    heavy_vehicles = class_counts.get('truck', 0) + class_counts.get('bus', 0)
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
    class_counts = {'car': 0, 'truck': 0}
    valid_detections = []

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywhn.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for bbox, cls, conf in zip(boxes, classes, confidences):
            cls_int = int(cls)
            class_name = 'car' if cls_int == 0 else 'truck'

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
    global models, model_scalers, feature_selector, feature_scaler

    predictions = {}

    feature_cols = [col for col in features_dict.keys() if col not in ['vehicle_count', 'traffic_label']]
    features_array = np.array([features_dict[col] for col in feature_cols]).reshape(1, -1)

    # Apply feature selector
    if feature_selector is not None:
        try:
            features_array = feature_selector.transform(features_array)
        except Exception as e:
            print(f"Feature selector error: {e}")
            pass

    # Make predictions with each model
    for model_name, model in models.items():
        try:
            # Apply model-specific scaler if available
            test_features = features_array.copy()

            if model_name in ['logistic_regression', 'mlp', 'svm']:
                scaler_key = f"{model_name}_scaler"
                if scaler_key in model_scalers:
                    test_features = model_scalers[scaler_key].transform(test_features)
            else:
                # Tree-based models don't need scaling, but check shape
                pass

            pred = model.predict(test_features)[0]
            if model_name == 'linear_regression':
                predictions['linear_regression'] = float(pred)
            else:
                predictions[model_name] = str(pred)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}, features shape: {features_array.shape}")
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

    if DEMO_MODE:
        return jsonify({"error": "Demo mode - use /demo/<image_name> instead"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        temp_path = '/tmp/temp_image.jpg'
        os.makedirs('/tmp', exist_ok=True)
        cv2.imwrite(temp_path, img)

        vehicle_count, class_counts, valid_detections, annotated_image = run_yolo_detection(temp_path)

        if annotated_image is None:
            return jsonify({"error": "YOLO detection failed"}), 500

        features = extract_all_features(valid_detections, class_counts, img.shape, ['car', 'truck'])
        predictions = get_predictions(features)
        traffic_label = classify_traffic_density(vehicle_count)

        _, img_buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        features_dict = {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v
                        for k, v in features.items()}

        response = {
            'status': 'success',
            'image': f'data:image/jpeg;base64,{img_base64}',
            'vehicle_count': vehicle_count,
            'car_count': class_counts.get('car', 0),
            'truck_count': class_counts.get('truck', 0),
            'traffic_label': traffic_label,
            'predictions': predictions,
            'features': features_dict
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/demo/<image_name>')
def demo(image_name):
    global sample_cache

    if image_name not in sample_cache:
        return jsonify({"error": "Demo image not found"}), 404

    cached = sample_cache[image_name]
    img_path = SAMPLE_IMAGES_DIR / cached.get('image_path', f'{image_name}.jpg')

    if img_path.exists():
        img = cv2.imread(str(img_path))
        _, img_buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        response = {
            'status': 'success',
            'image': f'data:image/jpeg;base64,{img_base64}',
            'vehicle_count': cached.get('vehicle_count', 0),
            'car_count': cached.get('car_count', 0),
            'truck_count': cached.get('truck_count', 0),
            'traffic_label': cached.get('traffic_label', 'Low'),
            'predictions': cached.get('predictions', {}),
            'features': cached.get('features', {})
        }
        return jsonify(response)

    return jsonify({"error": "Image file not found"}), 404

if __name__ == '__main__':
    print("🌐 Starting Flask server on http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host='localhost', port=5000)
