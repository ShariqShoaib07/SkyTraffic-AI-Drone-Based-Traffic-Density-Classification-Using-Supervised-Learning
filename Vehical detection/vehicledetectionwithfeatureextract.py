# vehicle_detection_with_feature_extraction.py
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

from ultralytics import YOLO
import cv2
import pandas as pd
import yaml
import glob
import torch
import gc
import numpy as np
from collections import Counter
import math
from sklearn.preprocessing import StandardScaler
import joblib

# USE_MIXED_PRECISION is defined before GPU initialization.

# -----------------------------
# GPU MEMORY OPTIMIZATION
# -----------------------------
print("🧹 Initializing GPU memory optimization...")
torch.cuda.empty_cache()
gc.collect()
torch.backends.cudnn.benchmark = True
if USE_MIXED_PRECISION:
    torch.set_float32_matmul_precision('high')

# -----------------------------
# CONFIGURATION
# -----------------------------
dataset_folder = r"D:\UNI\Sem6\Machine Learning\Project\YOLODataset"
image_folder = r"D:\UNI\Sem6\Machine Learning\Project\YOLODataset\val\images"
output_folder = r"D:\UNI\Sem6\Machine Learning\Project\Results"
annotated_folder = os.path.join(output_folder, "AnnotatedImages")
output_csv = os.path.join(output_folder, "traffic_labels.csv")
features_csv = os.path.join(output_folder, "ml_features.csv")
scaler_path = os.path.join(output_folder, "feature_scaler.pkl")

# OPTIMIZED GPU SETTINGS
TRAIN_EPOCHS = 15
TRAIN_IMGSZ = 640
TRAIN_BATCH = 8
TRAIN_DEVICE = 0
TRAIN_WORKERS = 0
TRAIN_PATIENCE = 15
USE_MIXED_PRECISION = True

# Vehicle classes
# class_0 = index 0 in your .txt files (104241 detections), class_1 = index 1 (17256 detections)
VEHICLE_CLASSES = ['vehicle_type_0', 'vehicle_type_1']
NUM_CLASSES = len(VEHICLE_CLASSES)

# DETECTION CONFIDENCE
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# TRAFFIC DENSITY THRESHOLDS
LOW_THRESHOLD = 5
MEDIUM_THRESHOLD = 20

# Ensure output dirs exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

# -----------------------------
# FEATURE EXTRACTION FUNCTIONS
# -----------------------------
def extract_spatial_features(valid_detections, img_shape):
    """
    Extract spatial distribution features from detected vehicles
    """
    features = {}
    
    if not valid_detections:
        # Return zeros if no detections
        features.update({
            'vehicle_count': 0,
            'density_score': 0.0,
            'spread_area': 0.0,
            'center_x_mean': 0.5,
            'center_y_mean': 0.5,
            'bbox_area_mean': 0.0,
            'bbox_area_std': 0.0,
            'aspect_ratio_mean': 0.0,
            'spatial_entropy': 0.0,
            'cluster_score': 0.0,
            'road_utilization': 0.0
        })
        return features
    
    img_height, img_width = img_shape[:2]
    vehicle_count = len(valid_detections)
    
    # Basic count feature
    features['vehicle_count'] = vehicle_count
    
    # Density score (normalized by image area)
    features['density_score'] = vehicle_count / (img_width * img_height) * 100000
    
    # Extract bounding box properties
    centers_x = []
    centers_y = []
    bbox_areas = []
    aspect_ratios = []
    
    for detection in valid_detections:
        bbox = detection['bbox']  # [x_center, y_center, width, height] in normalized coordinates
        
        center_x, center_y, width, height = bbox
        
        # Convert to pixel coordinates
        center_x_px = center_x * img_width
        center_y_px = center_y * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        centers_x.append(center_x_px)
        centers_y.append(center_y_px)
        
        bbox_area = width_px * height_px
        bbox_areas.append(bbox_area)
        
        aspect_ratio = width_px / height_px if height_px > 0 else 0
        aspect_ratios.append(aspect_ratio)
    
    # Spatial distribution features
    if centers_x:
        # Mean position
        features['center_x_mean'] = np.mean(centers_x) / img_width
        features['center_y_mean'] = np.mean(centers_y) / img_height
        
        # Bounding box statistics
        features['bbox_area_mean'] = np.mean(bbox_areas)
        features['bbox_area_std'] = np.std(bbox_areas) if len(bbox_areas) > 1 else 0.0
        features['aspect_ratio_mean'] = np.mean(aspect_ratios)
        
        # Spread area (convex hull area of all vehicle centers)
        if len(centers_x) >= 3:
            from scipy.spatial import ConvexHull
            points = np.column_stack([centers_x, centers_y])
            hull = ConvexHull(points)
            features['spread_area'] = hull.volume
        else:
            # For fewer points, use bounding rectangle area
            if centers_x:
                x_range = max(centers_x) - min(centers_x)
                y_range = max(centers_y) - min(centers_y)
                features['spread_area'] = x_range * y_range
            else:
                features['spread_area'] = 0.0
        
        # Spatial entropy (distribution randomness)
        spatial_entropy = calculate_spatial_entropy(centers_x, centers_y, img_width, img_height)
        features['spatial_entropy'] = spatial_entropy
        
        # Cluster score (how clustered vehicles are)
        features['cluster_score'] = calculate_cluster_score(centers_x, centers_y)
        
        # Road utilization (percentage of image width covered by vehicles)
        if centers_x:
            x_coverage = (max(centers_x) - min(centers_x)) / img_width
            features['road_utilization'] = x_coverage
        else:
            features['road_utilization'] = 0.0
    else:
        # Default values when no vehicles
        features.update({
            'center_x_mean': 0.5,
            'center_y_mean': 0.5,
            'bbox_area_mean': 0.0,
            'bbox_area_std': 0.0,
            'aspect_ratio_mean': 0.0,
            'spread_area': 0.0,
            'spatial_entropy': 0.0,
            'cluster_score': 0.0,
            'road_utilization': 0.0
        })
    
    return features

def calculate_spatial_entropy(centers_x, centers_y, img_width, img_height, grid_size=4):
    """
    Calculate spatial distribution entropy using grid-based approach
    """
    if not centers_x:
        return 0.0
    
    # Create grid
    grid_counts = np.zeros((grid_size, grid_size))
    
    for x, y in zip(centers_x, centers_y):
        grid_x = min(int(x / img_width * grid_size), grid_size - 1)
        grid_y = min(int(y / img_height * grid_size), grid_size - 1)
        grid_counts[grid_y, grid_x] += 1
    
    # Calculate probability distribution
    total_points = len(centers_x)
    probabilities = grid_counts / total_points
    
    # Calculate entropy
    entropy = 0.0
    for p in probabilities.flatten():
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by maximum possible entropy
    max_entropy = math.log2(grid_size * grid_size)
    return entropy / max_entropy if max_entropy > 0 else 0.0

def calculate_cluster_score(centers_x, centers_y):
    """
    Calculate how clustered the vehicles are (0 = dispersed, 1 = highly clustered)
    """
    if len(centers_x) < 2:
        return 0.0
    
    # Calculate mean distance to centroid
    centroid_x = np.mean(centers_x)
    centroid_y = np.mean(centers_y)
    
    distances = [math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2) 
                for x, y in zip(centers_x, centers_y)]
    
    mean_distance = np.mean(distances)
    
    # Calculate maximum possible distance (diagonal of bounding box)
    if centers_x:
        x_range = max(centers_x) - min(centers_x)
        y_range = max(centers_y) - min(centers_y)
        max_possible = math.sqrt(x_range**2 + y_range**2)
        
        # Cluster score: 1 - (mean_distance / max_possible)
        # Lower mean distance = more clustered
        if max_possible > 0:
            return 1 - (mean_distance / max_possible)
    
    return 0.0

def extract_vehicle_type_features(class_counts, actual_classes):
    """
    Extract features related to vehicle type distribution
    """
    features = {}
    total_vehicles = sum(class_counts.values())
    
    # Individual vehicle type counts
    for class_name in actual_classes:
        features[f'count_{class_name}'] = class_counts.get(class_name, 0)
    
    # Vehicle type ratios
    if total_vehicles > 0:
        for class_name in actual_classes:
            features[f'ratio_{class_name}'] = class_counts.get(class_name, 0) / total_vehicles
    else:
        for class_name in actual_classes:
            features[f'ratio_{class_name}'] = 0.0
    
    # Vehicle type diversity (entropy)
    type_entropy = 0.0
    if total_vehicles > 0:
        for count in class_counts.values():
            if count > 0:
                p = count / total_vehicles
                type_entropy -= p * math.log2(p)
    
    # Normalize by maximum entropy
    max_entropy = math.log2(len(actual_classes)) if len(actual_classes) > 0 else 1.0
    features['type_diversity'] = type_entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Heavy vehicle ratio (trucks + buses)
    heavy_vehicles = class_counts.get('truck', 0) + class_counts.get('bus', 0)
    features['heavy_vehicle_ratio'] = heavy_vehicles / total_vehicles if total_vehicles > 0 else 0.0
    
    return features

def extract_traffic_pattern_features(valid_detections, img_shape):
    """
    Extract traffic pattern and flow features
    """
    features = {}
    
    if not valid_detections:
        features.update({
            'lane_occupancy': 0.0,
            'traffic_flow_score': 0.0,
            'congestion_index': 0.0,
            'speed_estimate': 0.0
        })
        return features
    
    img_height, img_width = img_shape[:2]
    vehicle_count = len(valid_detections)
    
    # Lane occupancy estimation (simplified)
    y_positions = [detection['bbox'][1] * img_height for detection in valid_detections]
    if y_positions:
        lane_coverage = (max(y_positions) - min(y_positions)) / img_height
        features['lane_occupancy'] = lane_coverage
    else:
        features['lane_occupancy'] = 0.0
    
    # Traffic flow score (based on spatial distribution)
    x_positions = [detection['bbox'][0] * img_width for detection in valid_detections]
    if x_positions:
        x_variance = np.var(x_positions) / (img_width ** 2) if img_width > 0 else 0.0
        features['traffic_flow_score'] = 1.0 - min(x_variance * 10, 1.0)
    else:
        features['traffic_flow_score'] = 0.0
    
    # Congestion index (combination of density and distribution)
    density = vehicle_count / (img_width * img_height) * 100000
    if x_positions:
        spread = (max(x_positions) - min(x_positions)) / img_width
        features['congestion_index'] = density * (1 - spread)
    else:
        features['congestion_index'] = density
    
    # Speed estimate (simplified - based on vehicle types and distribution)
    motorcycle_count = sum(1 for d in valid_detections if d['class'] == 'motorcycle')
    # Very simplified speed estimation
    base_speed = 50  # km/h
    if vehicle_count > 0:
        speed_factor = 1.0 - (vehicle_count / 50)  # More vehicles = slower
        type_factor = 1.0 + (motorcycle_count / vehicle_count) * 0.5  # More motorcycles = slightly faster
        features['speed_estimate'] = base_speed * speed_factor * type_factor
    else:
        features['speed_estimate'] = base_speed
    
    return features

# -----------------------------
# DATASET ANALYSIS & FIXING
# -----------------------------
def analyze_dataset_classes(dataset_folder):
    """Comprehensive analysis of what's actually in the dataset"""
    print("🔍 Comprehensive Dataset Analysis...")
    
    train_label_files = glob.glob(os.path.join(dataset_folder, 'train', 'labels', '*.txt'))
    val_label_files = glob.glob(os.path.join(dataset_folder, 'val', 'labels', '*.txt'))
    
    train_class_ids = []
    val_class_ids = []
    
    for label_file in train_label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        train_class_ids.append(class_id)
        except (OSError, ValueError):
            continue
    
    for label_file in val_label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        val_class_ids.append(class_id)
        except (OSError, ValueError):
            continue
    
    train_counts = Counter(train_class_ids)
    val_counts = Counter(val_class_ids)
    
    print("📊 DATASET CLASS DISTRIBUTION:")
    print(f"   Training: {dict(train_counts)}")
    print(f"   Validation: {dict(val_counts)}")
    
    all_class_ids = set(train_class_ids + val_class_ids)
    print(f"   Unique class IDs: {sorted(all_class_ids)}")
    
    if len(all_class_ids) == 1 and 0 in all_class_ids:
        print("⚠️  ONLY CARS FOUND - Adjusting training strategy...")
        return ['car']
    else:
        return VEHICLE_CLASSES

def create_optimized_data_yaml(dataset_folder, classes):
    """Create optimized data.yaml for better training"""
    train_path = os.path.join(dataset_folder, 'train', 'images').replace("\\", "/")
    val_path = os.path.join(dataset_folder, 'val', 'images').replace("\\", "/")
    
    data_yaml = {
        'path': dataset_folder.replace("\\", "/"),
        'train': train_path,
        'val': val_path,
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(dataset_folder, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"✅ Created data.yaml with {len(classes)} classes: {classes}")
    return yaml_path

# -----------------------------
# MEMORY MANAGEMENT
# -----------------------------
def clear_gpu_memory():
    """Aggressive GPU memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"🧹 GPU Memory cleared - Current: {allocated:.2f}GB")

# -----------------------------
# ENHANCED VEHICLE DETECTION
# -----------------------------
def detect_all_vehicles(results, _img_shape, class_names):
    """Detect ALL vehicles without road filtering"""
    vehicle_count = 0
    class_counts = {class_name: 0 for class_name in class_names}
    valid_detections = []
    
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywhn.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for bbox, cls, conf in zip(boxes, classes, confidences):
            cls_int = int(cls)
            if cls_int < len(class_names):
                class_name = class_names[cls_int]
                
                if conf >= CONFIDENCE_THRESHOLD:
                    vehicle_count += 1
                    class_counts[class_name] += 1
                    valid_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': bbox
                    })
    
    return vehicle_count, class_counts, valid_detections

# -----------------------------
# TRAFFIC DENSITY CLASSIFICATION
# -----------------------------
def classify_traffic_density(vehicle_count):
    """Traffic density classification"""
    if vehicle_count <= LOW_THRESHOLD:
        return "Low"
    elif vehicle_count <= MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"

# -----------------------------
# FEATURE EXTRACTION PIPELINE
# -----------------------------
def extract_all_features(valid_detections, class_counts, img_shape, actual_classes):
    """
    Extract comprehensive features for ML classifier
    """
    features = {}
    
    # 1. Basic count features
    features['vehicle_count'] = len(valid_detections)
    
    # 2. Spatial distribution features
    spatial_features = extract_spatial_features(valid_detections, img_shape)
    features.update(spatial_features)
    
    # 3. Vehicle type features
    type_features = extract_vehicle_type_features(class_counts, actual_classes)
    features.update(type_features)
    
    # 4. Traffic pattern features
    pattern_features = extract_traffic_pattern_features(valid_detections, img_shape)
    features.update(pattern_features)
    
    # 5. Confidence statistics
    if valid_detections:
        confidences = [d['confidence'] for d in valid_detections]
        features['confidence_mean'] = np.mean(confidences)
        features['confidence_std'] = np.std(confidences) if len(confidences) > 1 else 0.0
    else:
        features['confidence_mean'] = 0.0
        features['confidence_std'] = 0.0
    
    return features

# -----------------------------
# MAIN EXECUTION WITH FEATURE EXTRACTION
# -----------------------------
def main():
    print("🚗 ENHANCED VEHICLE DETECTION WITH FEATURE EXTRACTION")
    print("=" * 60)
    print("🎯 PHASE 6: Feature Extraction for ML Classifier")
    print("=" * 60)
    
    # GPU initialization
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎯 GPU: {device_name}")
        print(f"🎯 VRAM: {vram:.1f} GB")
    else:
        print("❌ No GPU available!")
        return
    
    # Analyze dataset and determine actual classes
    actual_classes = analyze_dataset_classes(dataset_folder)
    print(f"🎯 Training with classes: {actual_classes}")
    
    # Create optimized data configuration
    data_yaml_path = create_optimized_data_yaml(dataset_folder, actual_classes)
    print(f"ðŸ“ Data configuration: {data_yaml_path}")
    
    # Load appropriate model
    if len(actual_classes) == 1:
        print("🎯 Using YOLOv8s (single class optimization)")
        model = YOLO("yolov8s.pt")
    else:
        print("🎯 Using YOLOv8m (multi-class optimization)")
        model = YOLO("yolov8m.pt")
    
    # Train model (you can skip this if you already have a trained model)
    try:
        # Uncomment to train new model
        # model.train(data=data_yaml_path, epochs=TRAIN_EPOCHS, imgsz=TRAIN_IMGSZ, 
        #            batch=TRAIN_BATCH, device=TRAIN_DEVICE, workers=TRAIN_WORKERS)
        print("✅ Using existing trained model...")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Load best model
    best_weights = find_best_pt()
    if not best_weights:
        print("❌ No trained model found!")
        return
    
    print(f"✅ Loading best model: {os.path.basename(best_weights)}")
    model = YOLO(best_weights)
    
    # Process images and extract features
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"\n🔮 PROCESSING {len(image_files)} IMAGES WITH FEATURE EXTRACTION...")
    
    # Initialize results storage
    all_features = []
    data = {"image_name": [], "vehicle_count": [], "traffic_label": []}
    
    for img_idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Could not load: {img_name}")
            continue
        
        # Clear memory periodically
        if img_idx % 10 == 0:
            clear_gpu_memory()
        
        try:
            # Prediction
            results = model.predict(
                img_path,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=TRAIN_IMGSZ,
                max_det=100,
                device=TRAIN_DEVICE,
                verbose=False
            )
            
            # Vehicle detection
            vehicle_count, class_counts, valid_detections = detect_all_vehicles(
                results, img.shape, actual_classes
            )
            
            # 🟥 PHASE 6: EXTRACT ML FEATURES
            features = extract_all_features(valid_detections, class_counts, img.shape, actual_classes)
            features['image_name'] = img_name
            
            # Traffic density classification
            label = classify_traffic_density(vehicle_count)
            features['traffic_label'] = label
            
            # Store features
            all_features.append(features)
            
            # Store basic results
            data["image_name"].append(img_name)
            data["vehicle_count"].append(vehicle_count)
            data["traffic_label"].append(label)
            
            # Progress reporting
            print(f"   [{img_idx:02d}/{len(image_files):02d}] {img_name}:")
            print(f"        🚗 Vehicles: {vehicle_count} - Label: {label}")
            print(f"        📊 Features: {len(features)} extracted")
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            continue
    
    # Save comprehensive feature dataset
    if all_features:
        features_df = pd.DataFrame(all_features)
        
        # Separate features and target for ML
        feature_columns = [col for col in features_df.columns if col not in ['image_name', 'traffic_label']]
        # Standardize features (important for ML models)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df[feature_columns])
        
        # Create final ML-ready dataset
        ml_dataset = pd.DataFrame(features_scaled, columns=feature_columns)
        ml_dataset['image_name'] = features_df['image_name']
        ml_dataset['traffic_label'] = features_df['traffic_label']
        
        # Save datasets
        features_df.to_csv(features_csv, index=False)
        ml_dataset.to_csv(os.path.join(output_folder, "ml_ready_dataset.csv"), index=False)
        joblib.dump(scaler, scaler_path)
        
        print(f"\n✅ FEATURE EXTRACTION COMPLETED!")
        print(f"📊 Total features extracted per image: {len(feature_columns)}")
        print(f"📁 Feature files saved:")
        print(f"   - Raw features: {features_csv}")
        print(f"   - ML-ready dataset: {output_folder}/ml_ready_dataset.csv")
        print(f"   - Feature scaler: {scaler_path}")
        
        # Display feature summary
        print(f"\n🎯 FEATURE SUMMARY:")
        print(f"   Basic Count Features: vehicle_count")
        print(f"   Spatial Features: density_score, spread_area, spatial_entropy, etc.")
        print(f"   Vehicle Type Features: count_*, ratio_*, type_diversity")
        print(f"   Traffic Pattern Features: lane_occupancy, congestion_index, speed_estimate")
        print(f"   Confidence Features: confidence_mean, confidence_std")
        
        # Show sample of extracted features
        print(f"\n📋 SAMPLE FEATURES (first image):")
        sample_features = {k: v for k, v in all_features[0].items() if k not in ['image_name', 'traffic_label']}
        for key, value in list(sample_features.items())[:10]:  # Show first 10 features
            print(f"   {key}: {value:.4f}")
    
    # Save basic results
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"\n🎉 PHASE 6 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("🟥 NEXT STEPS:")
    print("   1. Use ml_ready_dataset.csv for ML model training")
    print("   2. Train Random Forest / SVM / CNN classifier")
    print("   3. Evaluate model performance")
    print("   4. Create statistical visualizations")

def find_best_pt():
    """Find the best trained model"""
    candidates = glob.glob(os.path.join("runs", "**", "weights", "best.pt"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join("runs", "**", "best.pt"), recursive=True)
    
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p))
        return candidates[-1]
    return None

if __name__ == '__main__':
    main()

