# vehicle_detection_optimized_fixed.py
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

from ultralytics import YOLO
import cv2
import pandas as pd
import yaml
import glob
import torch
import gc
from collections import Counter

USE_MIXED_PRECISION = True


# -----------------------------
# GPU MEMORY OPTIMIZATION
# -----------------------------
def initialize_gpu_memory():
    """Initialize GPU settings only in the main process."""
    print("🧹 Initializing GPU memory optimization...")
    if torch.cuda.is_available():
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

# OPTIMIZED GPU SETTINGS - RTX 4060 (8GB VRAM)
TRAIN_MODEL = False            # Set True only when you want to train again
TRAIN_EPOCHS = 15
TRAIN_IMGSZ = 640              
TRAIN_BATCH = 8                # Safer for RTX 4060 Laptop 8GB VRAM
TRAIN_DEVICE = 0               
TRAIN_WORKERS = 0              # Avoid Windows worker re-import issues
TRAIN_PATIENCE = 15
# USE_MIXED_PRECISION is defined before GPU initialization.

# Vehicle classes - FOCUS ON ROAD VEHICLES ONLY
# UPDATE THESE NAMES to match your dataset's actual class labels
# class_0 = index 0 in your .txt files (104241), class_1 = index 1 (17256)
VEHICLE_CLASSES = ['vehicle_type_0', 'vehicle_type_1']
NUM_CLASSES = len(VEHICLE_CLASSES)

# DETECTION CONFIDENCE - LOWER FOR BETTER DETECTION
CONFIDENCE_THRESHOLD = 0.25    # LOWER confidence to detect more vehicles
IOU_THRESHOLD = 0.45           # Better overlap handling

# TRAFFIC DENSITY THRESHOLDS - ADJUSTED FOR BETTER DISTRIBUTION
LOW_THRESHOLD = 5              # 0-5 vehicles = Low
MEDIUM_THRESHOLD = 20          # 6-20 vehicles = Medium
# Above 20 = High

# Ensure output dirs exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

# -----------------------------
# DATASET ANALYSIS & FIXING
# -----------------------------
def analyze_dataset_classes(dataset_folder):
    """
    Comprehensive analysis of what's actually in the dataset
    """
    print("🔍 Comprehensive Dataset Analysis...")
    
    train_label_files = glob.glob(os.path.join(dataset_folder, 'train', 'labels', '*.txt'))
    val_label_files = glob.glob(os.path.join(dataset_folder, 'val', 'labels', '*.txt'))
    
    train_class_ids = []
    val_class_ids = []
    
    # Analyze training labels
    for idx, label_file in enumerate(train_label_files, 1):
        try:
            with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Valid YOLO format
                        class_id = int(float(parts[0]))
                        train_class_ids.append(class_id)
            if idx % 2000 == 0:
                print(f"   Scanned {idx}/{len(train_label_files)} training label files...")
        except (OSError, ValueError) as exc:
            print(f"   Skipping invalid training label {label_file}: {exc}")
            continue
    
    # Analyze validation labels
    for idx, label_file in enumerate(val_label_files, 1):
        try:
            with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        val_class_ids.append(class_id)
            if idx % 1000 == 0:
                print(f"   Scanned {idx}/{len(val_label_files)} validation label files...")
        except (OSError, ValueError) as exc:
            print(f"   Skipping invalid validation label {label_file}: {exc}")
            continue
    
    # Count class distributions
    train_counts = Counter(train_class_ids)
    val_counts = Counter(val_class_ids)
    
    print("📊 DATASET CLASS DISTRIBUTION:")
    print(f"   Training: {dict(train_counts)}")
    print(f"   Validation: {dict(val_counts)}")
    
    # Determine actual classes present
    all_class_ids = set(train_class_ids + val_class_ids)
    print(f"   Unique class IDs: {sorted(all_class_ids)}")
    
    # If only cars are present, we'll focus on that
    if len(all_class_ids) == 1 and 0 in all_class_ids:
        print("⚠️  ONLY CARS FOUND - Adjusting training strategy...")
        return ['car']  # Focus only on cars
    else:
        return VEHICLE_CLASSES  # Use all expected classes

def create_optimized_data_yaml(dataset_folder, classes):
    """
    Create optimized data.yaml for better training
    """
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
# ENHANCED VEHICLE DETECTION - NO ROAD FILTERING
# -----------------------------
def detect_all_vehicles(results, _img_shape, class_names):
    """
    Detect ALL vehicles without road filtering
    """
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
                
                # ONLY CONFIDENCE FILTERING - NO ROAD/SIZE/ASPECT FILTERS
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
# OPTIMIZED TRAINING FOR BETTER CLASS DISCRIMINATION
# -----------------------------
def train_optimized(model, data_yaml_path):
    """
    Optimized training for better vehicle type discrimination
    """
    print("🚀 Starting optimized training for vehicle detection...")
    
    clear_gpu_memory()
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=TRAIN_EPOCHS,
            imgsz=TRAIN_IMGSZ,
            batch=TRAIN_BATCH,
            device=TRAIN_DEVICE,
            workers=TRAIN_WORKERS,
            patience=TRAIN_PATIENCE,
            
            # Enhanced learning parameters
            lr0=0.01,           # Learning rate
            lrf=0.01,           # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # OPTIMIZED loss weights - HIGHER CLASS LOSS for better discrimination
            box=7.5,            # Box loss gain
            cls=1.0,            # INCREASED class loss for better vehicle type recognition
            dfl=1.5,            # Distribution Focal Loss
            
            # Enhanced augmentation for vehicles
            hsv_h=0.015,        # Color augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,        # No rotation (vehicles should be upright)
            translate=0.1,      # Translation
            scale=0.2,          # Scale augmentation
            shear=0.0,          # No shear
            perspective=0.0001, # Minimal perspective
            flipud=0.0,         # No vertical flip
            fliplr=0.5,         # Horizontal flip
            mosaic=1.0,         # Mosaic augmentation
            mixup=0.0,          # NO mixup to prevent class confusion
            
            save=True,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            single_cls=False,    # Multi-class training
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            # Add class weights if dataset is imbalanced
            # class_weights=[1.0, 1.5, 1.5, 2.0, 1.5]  # Higher weights for rare classes
        )
        return results
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU Memory Error! Using ultra-safe settings...")
        clear_gpu_memory()
        
        return model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=480,
            batch=4,
            device=TRAIN_DEVICE,
            workers=2,
            patience=15,
            save=True,
            exist_ok=True
        )

# -----------------------------
# FIND BEST MODEL
# -----------------------------
def find_best_pt():
    """Find the best trained model"""
    candidates = glob.glob(os.path.join("runs", "**", "weights", "best.pt"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join("runs", "**", "best.pt"), recursive=True)
    
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p))
        return candidates[-1]
    return None

# -----------------------------
# ENHANCED VISUALIZATION - NO ROAD MARKINGS
# -----------------------------
def create_enhanced_annotation(results, img, valid_detections):
    """
    Create enhanced visualization WITHOUT road region
    """
    annotated_img = results[0].plot() if results else img.copy()
    # Add detection statistics
    total_vehicles = len(valid_detections)
    stats_text = f"Total Vehicles: {total_vehicles}"
    cv2.putText(annotated_img, stats_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add class breakdown for top 3 classes
    class_counts = {}
    for detection in valid_detections:
        cls = detection['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (cls, count) in enumerate(sorted_classes[:3]):
        y_pos = 60 + i * 25
        text = f"{cls}: {count}"
        cv2.putText(annotated_img, text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return annotated_img

# -----------------------------
# IMPROVED TRAFFIC DENSITY CLASSIFICATION
# -----------------------------
def classify_traffic_density(vehicle_count):
    """
    Improved traffic density classification with better thresholds
    """
    if vehicle_count <= LOW_THRESHOLD:
        return "Low"
    elif vehicle_count <= MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    initialize_gpu_memory()
    print("🚗 ENHANCED VEHICLE DETECTION PIPELINE")
    print("=" * 60)
    print("🎯 Focus: ALL VEHICLES - No road filtering")
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
    
    if TRAIN_MODEL:
        # Load appropriate model based on dataset
        if len(actual_classes) == 1:
            print("🎯 Using YOLOv8s (single class optimization)")
            model = YOLO("yolov8s.pt")
        else:
            print("🎯 Using YOLOv8m (multi-class optimization)")
            model = YOLO("yolov8m.pt")
        
        print(f"\n🔥 OPTIMIZED TRAINING CONFIGURATION:")
        print(f"   Epochs: {TRAIN_EPOCHS}")
        print(f"   Image Size: {TRAIN_IMGSZ}")
        print(f"   Batch Size: {TRAIN_BATCH}")
        print(f"   Workers: {TRAIN_WORKERS}")
        print(f"   Classes: {actual_classes}")
        print(f"   Confidence: {CONFIDENCE_THRESHOLD}")
        print(f"   Traffic Thresholds: Low<={LOW_THRESHOLD}, Medium<={MEDIUM_THRESHOLD}, High>{MEDIUM_THRESHOLD}")
        
        # Optimized training
        try:
            train_optimized(model, data_yaml_path)
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return
    else:
        print("⏭️  Skipping training - using existing best.pt")
    
    # Load best model
    best_weights = find_best_pt()
    if not best_weights:
        print("❌ No trained model found!")
        return
    
    print(f"✅ Loading best model: {os.path.basename(best_weights)}")
    model = YOLO(best_weights)
    
    # Enhanced inference - NO ROAD FILTERING
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"\n🔮 PROCESSING {len(image_files)} IMAGES (NO ROAD FILTERING)...")
    
    # Initialize results storage
    data = {"image_name": [], "vehicle_count": [], "traffic_label": []}
    for class_name in actual_classes:
        data[class_name] = []
    
    total_vehicles = 0
    detection_stats = {class_name: 0 for class_name in actual_classes}
    traffic_distribution = {"Low": 0, "Medium": 0, "High": 0}
    
    for img_idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Could not load: {img_name}")
            continue
        
        # Clear memory periodically
        if img_idx % 10 == 0:
            clear_gpu_memory()
        
        # Enhanced prediction WITHOUT road filtering
        try:
            results = model.predict(
                img_path,
                conf=CONFIDENCE_THRESHOLD,  # LOWER CONFIDENCE for more detections
                iou=IOU_THRESHOLD,
                imgsz=TRAIN_IMGSZ,
                max_det=100,  # INCREASED max detections
                device=TRAIN_DEVICE,
                verbose=False
            )
            
            # Apply detection WITHOUT road filtering
            vehicle_count, class_counts, valid_detections = detect_all_vehicles(
                results, img.shape, actual_classes
            )
            
            # Update statistics
            total_vehicles += vehicle_count
            for class_name, count in class_counts.items():
                detection_stats[class_name] += count
            
            # IMPROVED traffic density classification
            label = classify_traffic_density(vehicle_count)
            traffic_distribution[label] += 1
            
            # Store results
            data["image_name"].append(img_name)
            data["vehicle_count"].append(vehicle_count)
            data["traffic_label"].append(label)
            for class_name in actual_classes:
                data[class_name].append(class_counts.get(class_name, 0))
            
            # Create enhanced visualization WITHOUT road markings
            annotated_img = create_enhanced_annotation(results, img, valid_detections)
            save_path = os.path.join(annotated_folder, f"detected_{img_name}")
            cv2.imwrite(save_path, annotated_img)
            
            # Progress reporting
            print(f"   [{img_idx:02d}/{len(image_files):02d}] {img_name}:")
            print(f"        🚗 Total Vehicles: {vehicle_count} - {label}")
            if vehicle_count > 0:
                details = [f"{k}:{v}" for k, v in class_counts.items() if v > 0]
                print(f"        📊 Breakdown: {', '.join(details)}")
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            continue
    
    # Save comprehensive results
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    # Final statistics
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 FINAL VEHICLE DETECTION STATISTICS:")
    print(f"   Total Images Processed: {len(image_files)}")
    print(f"   Total Vehicles Detected: {total_vehicles}")
    print(f"   Average Vehicles per Image: {total_vehicles/len(image_files):.1f}")
    print(f"   Vehicle Type Distribution:")
    for class_name in actual_classes:
        count = detection_stats[class_name]
        if count > 0:
            percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
            print(f"        {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n🚦 TRAFFIC DENSITY DISTRIBUTION:")
    for label in ['Low', 'Medium', 'High']:
        count = traffic_distribution[label]
        percentage = (count / len(image_files) * 100) if len(image_files) > 0 else 0
        print(f"   {label} Traffic: {count} images ({percentage:.1f}%)")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   CSV Results: {output_csv}")
    print(f"   Annotated Images: {annotated_folder}")

if __name__ == '__main__':
    main()

