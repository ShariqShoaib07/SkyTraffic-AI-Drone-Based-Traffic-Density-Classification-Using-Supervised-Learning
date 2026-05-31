# Project Title
SkyTraffic AI - YOLO-Based Vehicle Detection and Traffic Density Classification

## Overview
This project builds an end-to-end pipeline for vehicle detection and traffic density classification from UAV/drone aerial images. It trains or loads a YOLO detector, counts vehicles per image, and assigns each scene a Low, Medium, or High traffic label. Those labels and detections feed a classical ML classifier that predicts traffic density from engineered features.

The raw dataset lives under [dataset/dataset](dataset/dataset) and is organized into up to 12 sections (sec1–sec9, sec_a, sec_b, sec_c) with YOLO-format label files that store class_id cx cy width height values in normalized coordinates. The dataset contains approximately 15,000 images (15,070 total), with each image paired to a TXT label file containing bounding boxes and class IDs. Images are Spanish road traffic scenes captured from UAVs (drones).

**Dataset Classes:**
- Class 0 (cars): 137,602 detections  
- Class 1 (motorcycles): 17,726 detections  
- Total labeled vehicles: 155,328  
- Ratio: ~88% cars, ~12% motorcycles

The pipeline restructures the raw dataset into [YOLODataset](YOLODataset) train and val splits, performs YOLO inference to produce [Results/traffic_labels.csv](Results/traffic_labels.csv) and annotated images in [Results/AnnotatedImages](Results/AnnotatedImages), extracts spatial and traffic-pattern features into [Results/ml_features.csv](Results/ml_features.csv) and [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv), trains Random Forest, SVM, and MLP classifiers, evaluates them statistically, and runs a research-inspired advanced analysis that adds particle-filter and neural Kalman filter features.

## Project Structure
- [inspect_dataset.py](inspect_dataset.py) - Scans sec1-sec9, sec_a-sec_c for class ID counts and prints sample image names.
- [restructure_dataset.py](restructure_dataset.py) - Copies PNG and TXT pairs into [YOLODataset](YOLODataset) train and val splits.
- [Vehical detection/VehicleDetectionAndLabeling.py](Vehical%20detection/VehicleDetectionAndLabeling.py) - Trains a YOLO detector and generates [Results/traffic_labels.csv](Results/traffic_labels.csv) plus annotated images in [Results/AnnotatedImages](Results/AnnotatedImages).
- [Vehical detection/vehicledetectionwithfeatureextract.py](Vehical%20detection/vehicledetectionwithfeatureextract.py) - Runs YOLO inference and extracts ML features into [Results/ml_features.csv](Results/ml_features.csv) and [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv).
- [Vehical detection/mlclassifiertraining.py](Vehical%20detection/mlclassifiertraining.py) - Trains Random Forest, SVM, and MLP classifiers, saving models in [Results/trained_models](Results/trained_models) and plots in [Results/ml_results](Results/ml_results).
- [Vehical detection/statisticalanalysisandevaluation.py](Vehical%20detection/statisticalanalysisandevaluation.py) - Produces statistical plots and evaluation reports in [Results/statistical_analysis](Results/statistical_analysis).
- [Vehical detection/advancedtrafficanalysiswithresearch.py](Vehical%20detection/advancedtrafficanalysiswithresearch.py) - Generates research-inspired advanced features in [Results/advanced_analysis](Results/advanced_analysis).
- [Vehical detection/VehicleDetectionAndLabeling.pyproj](Vehical%20detection/VehicleDetectionAndLabeling.pyproj) - Visual Studio Python project metadata for the Vehical detection folder.
- [yolo11n.pt](yolo11n.pt) - YOLO weight file stored in the project root.
- [yolo26n.pt](yolo26n.pt) - YOLO weight file stored in the project root.
- [yolov8m.pt](yolov8m.pt) - YOLO weight file stored in the project root (used as a base model in training).
- [dataset](dataset) - Raw dataset root containing scenes.csv and dataset subfolders.
- [dataset/dataset](dataset/dataset) - Raw dataset sections folder that holds sec1-sec4 (and any additional sections used in restructuring).
- [dataset/scenes.csv](dataset/scenes.csv) - Raw dataset metadata CSV.
- [YOLODataset](YOLODataset) - YOLO-formatted dataset with train and val splits.
- [YOLODataset/data.yaml](YOLODataset/data.yaml) - Dataset configuration file written by the detection scripts.
- [Results](Results) - Output root for CSVs, models, and analysis artifacts.
- [runs](runs) - Ultralytics training runs and checkpoint outputs.
- [pipeline_run.log](pipeline_run.log) - Captured pipeline stdout log.
- [pipeline_run.err.log](pipeline_run.err.log) - Captured pipeline stderr log.
- [archive.zip](archive.zip) - Zip archive stored in the project root.
- [ML project.zip](ML%20project.zip) - Zip archive stored in the project root.
- [SkyTraffic_AI_Proposal.docx.pdf](SkyTraffic_AI_Proposal.docx.pdf) - Project proposal document.
- [.venv](.venv) - Local Python virtual environment folder.
- [.vscode](.vscode) - Workspace editor settings.
- [__pycache__](__pycache__) - Python bytecode cache folder.

## Pipeline — How It Works
1. Dataset restructuring: [restructure_dataset.py](restructure_dataset.py) reads raw PNG and TXT pairs from D:\UNI\Sem6\Machine Learning\Project\dataset\dataset\sec1-sec7 (if present) and writes [YOLODataset/train/images](YOLODataset/train/images), [YOLODataset/train/labels](YOLODataset/train/labels), [YOLODataset/val/images](YOLODataset/val/images), and [YOLODataset/val/labels](YOLODataset/val/labels) under D:\UNI\Sem6\Machine Learning\Project\YOLODataset.
2. YOLO detection + labeling: [Vehical detection/VehicleDetectionAndLabeling.py](Vehical%20detection/VehicleDetectionAndLabeling.py) creates [YOLODataset/data.yaml](YOLODataset/data.yaml), trains a YOLOv8s or YOLOv8m detector, and runs inference on D:\UNI\Sem6\Machine Learning\Project\YOLODataset\val\images to produce [Results/traffic_labels.csv](Results/traffic_labels.csv) and annotated images in [Results/AnnotatedImages](Results/AnnotatedImages).
3. Feature extraction: [Vehical detection/vehicledetectionwithfeatureextract.py](Vehical%20detection/vehicledetectionwithfeatureextract.py) loads the latest YOLO checkpoint from [runs](runs), extracts spatial and traffic features per image, and writes [Results/ml_features.csv](Results/ml_features.csv), [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv), [Results/feature_scaler.pkl](Results/feature_scaler.pkl), and [Results/traffic_labels.csv](Results/traffic_labels.csv).
4. ML classifier training: [Vehical detection/mlclassifiertraining.py](Vehical%20detection/mlclassifiertraining.py) trains Random Forest, SVM, and MLP classifiers on [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv) (or [Results/ml_features.csv](Results/ml_features.csv)) and saves trained models in [Results/trained_models](Results/trained_models) plus plots and summaries in [Results/ml_results](Results/ml_results).
5. Statistical evaluation: [Vehical detection/statisticalanalysisandevaluation.py](Vehical%20detection/statisticalanalysisandevaluation.py) evaluates the trained models and generates plots and reports in [Results/statistical_analysis](Results/statistical_analysis).
6. Advanced analysis: [Vehical detection/advancedtrafficanalysiswithresearch.py](Vehical%20detection/advancedtrafficanalysiswithresearch.py) adds research-inspired features from [Results/ml_features.csv](Results/ml_features.csv) and writes advanced outputs in [Results/advanced_analysis](Results/advanced_analysis).

## Models Used
- `YOLOv8s` and `YOLOv8m` (Ultralytics) - Object detectors used to localize vehicles and create per-image counts.
- `RandomForestClassifier` - Ensemble classifier for traffic density using engineered features.
- `SVC` (RBF kernel) - Support Vector Machine classifier for traffic density.
- `MLPClassifier` - Multi-layer perceptron neural network classifier for traffic density.
- `KNeighborsClassifier` - K-Nearest Neighbors classifier for traffic density classification.
- `LogisticRegression` - Logistic regression classifier for multi-class traffic density prediction.
- `DecisionTreeClassifier` - Decision tree classifier with tree visualization and rule extraction.
- `LinearRegression` - Regression model to predict vehicle count as continuous value, feeding density classification.
- `TrafficParticleFilter` - PoseRBPF-inspired particle filter for traffic state estimation in advanced analysis.
- `NeuralKalmanFilter` - KalmanNet-inspired neural Kalman filter for state estimation in advanced analysis.
- `DBSCAN` - Clustering algorithm used to compute vehicle cluster density in advanced features.

## Features Extracted
[Results/ml_features.csv](Results/ml_features.csv) contains image_name and traffic_label plus the features below.
- `vehicle_count`
- `density_score`
- `spread_area`
- `center_x_mean`
- `center_y_mean`
- `bbox_area_mean`
- `bbox_area_std`
- `aspect_ratio_mean`
- `spatial_entropy`
- `cluster_score`
- `road_utilization`
- `count_<class_name>` for each class in `actual_classes`
- `ratio_<class_name>` for each class in `actual_classes`
- `type_diversity`
- `motorcycle_ratio`
- `lane_occupancy`
- `traffic_flow_score`
- `congestion_index`
- `speed_estimate`
- `confidence_mean`
- `confidence_std`

## Output Files
- **YOLO training and labeling**: [YOLODataset/data.yaml](YOLODataset/data.yaml), [runs](runs), [Results/traffic_labels.csv](Results/traffic_labels.csv), [Results/AnnotatedImages](Results/AnnotatedImages).
- **Feature extraction**: [Results/ml_features.csv](Results/ml_features.csv), [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv), [Results/feature_scaler.pkl](Results/feature_scaler.pkl).
- **ML training outputs**: [Results/ml_results/feature_importance.png](Results/ml_results/feature_importance.png), [Results/ml_results/confusion_matrix_random_forest.png](Results/ml_results/confusion_matrix_random_forest.png), [Results/ml_results/confusion_matrix_svm.png](Results/ml_results/confusion_matrix_svm.png), [Results/ml_results/confusion_matrix_mlp.png](Results/ml_results/confusion_matrix_mlp.png), [Results/ml_results/confusion_matrix_knn.png](Results/ml_results/confusion_matrix_knn.png), [Results/ml_results/confusion_matrix_logistic_regression.png](Results/ml_results/confusion_matrix_logistic_regression.png), [Results/ml_results/confusion_matrix_decision_tree.png](Results/ml_results/confusion_matrix_decision_tree.png), [Results/ml_results/decision_tree_rules.txt](Results/ml_results/decision_tree_rules.txt), [Results/ml_results/decision_tree_plot.png](Results/ml_results/decision_tree_plot.png), [Results/ml_results/linear_regression_predictions.png](Results/ml_results/linear_regression_predictions.png), [Results/ml_results/feature_importance_random_forest.png](Results/ml_results/feature_importance_random_forest.png), [Results/ml_results/model_comparison_chart.png](Results/ml_results/model_comparison_chart.png), [Results/ml_results/training_summary.json](Results/ml_results/training_summary.json), [Results/ml_results/training_summary.txt](Results/ml_results/training_summary.txt).
- **Saved models and selectors**: [Results/trained_models/random_forest_model.pkl](Results/trained_models/random_forest_model.pkl), [Results/trained_models/svm_model.pkl](Results/trained_models/svm_model.pkl), [Results/trained_models/mlp_model.pkl](Results/trained_models/mlp_model.pkl), [Results/trained_models/knn_model.pkl](Results/trained_models/knn_model.pkl), [Results/trained_models/logistic_regression_model.pkl](Results/trained_models/logistic_regression_model.pkl), [Results/trained_models/decision_tree_model.pkl](Results/trained_models/decision_tree_model.pkl), [Results/trained_models/linear_regression_model.pkl](Results/trained_models/linear_regression_model.pkl), [Results/trained_models/svm_scaler.pkl](Results/trained_models/svm_scaler.pkl), [Results/trained_models/mlp_scaler.pkl](Results/trained_models/mlp_scaler.pkl), [Results/trained_models/lr_scaler.pkl](Results/trained_models/lr_scaler.pkl), [Results/trained_models/feature_selector.pkl](Results/trained_models/feature_selector.pkl), [Results/trained_models/selected_features.pkl](Results/trained_models/selected_features.pkl).
- **Statistical analysis**: [Results/statistical_analysis/data_distribution_analysis.png](Results/statistical_analysis/data_distribution_analysis.png), [Results/statistical_analysis/feature_analysis.png](Results/statistical_analysis/feature_analysis.png), [Results/statistical_analysis/comprehensive_model_comparison.png](Results/statistical_analysis/comprehensive_model_comparison.png), [Results/statistical_analysis/model_performance_metrics.csv](Results/statistical_analysis/model_performance_metrics.csv), [Results/statistical_analysis/statistical_analysis_report.txt](Results/statistical_analysis/statistical_analysis_report.txt), [Results/statistical_analysis/best_model_confusion_matrix.png](Results/statistical_analysis/best_model_confusion_matrix.png).
- **Advanced analysis**: [Results/advanced_analysis/advanced_features.csv](Results/advanced_analysis/advanced_features.csv), [Results/advanced_analysis/research_integration_report.txt](Results/advanced_analysis/research_integration_report.txt), [Results/advanced_analysis/research_integration_analysis.png](Results/advanced_analysis/research_integration_analysis.png).

## Setup & Installation
```bash
# Clone the repo
git clone <your-repo-url>
cd <project-folder>

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install --no-cache-dir ultralytics opencv-python pandas numpy pyyaml torch torchvision scikit-learn joblib matplotlib seaborn
```

## How To Run
1. Dataset restructuring.
```bash
python restructure_dataset.py
```
Expect [YOLODataset/train](YOLODataset/train) and [YOLODataset/val](YOLODataset/val) to be created and populated.

2. YOLO detection and labeling.
```bash
python "Vehical detection/VehicleDetectionAndLabeling.py"
```
Expect [Results/traffic_labels.csv](Results/traffic_labels.csv), [Results/AnnotatedImages](Results/AnnotatedImages), and new training runs under [runs](runs).

3. Feature extraction.
```bash
python "Vehical detection/vehicledetectionwithfeatureextract.py"
```
Expect [Results/ml_features.csv](Results/ml_features.csv), [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv), and [Results/feature_scaler.pkl](Results/feature_scaler.pkl).

4. ML classifier training.
```bash
python "Vehical detection/mlclassifiertraining.py"
```
Expect trained models in [Results/trained_models](Results/trained_models) and plots plus summaries in [Results/ml_results](Results/ml_results).

5. Statistical evaluation.
```bash
python "Vehical detection/statisticalanalysisandevaluation.py"
```
Expect plots and reports under [Results/statistical_analysis](Results/statistical_analysis).

6. Advanced analysis.
```bash
python "Vehical detection/advancedtrafficanalysiswithresearch.py"
```
Expect [Results/advanced_analysis/advanced_features.csv](Results/advanced_analysis/advanced_features.csv) and research reports under [Results/advanced_analysis](Results/advanced_analysis).

## Dataset Format
Labels follow the YOLO format: class_id cx cy width height with all coordinates normalized to 0-1 relative to image size. After running [restructure_dataset.py](restructure_dataset.py), the dataset is organized as [YOLODataset/train/images](YOLODataset/train/images), [YOLODataset/train/labels](YOLODataset/train/labels), [YOLODataset/val/images](YOLODataset/val/images), and [YOLODataset/val/labels](YOLODataset/val/labels). The raw dataset is stored in D:\UNI\Sem6\Machine Learning\Project\dataset\dataset (workspace path [dataset/dataset](dataset/dataset)) with up to 12 sections of UAV traffic images: sec1 through sec9, sec_a, sec_b, sec_c.

## Configuration
- [inspect_dataset.py](inspect_dataset.py): `SOURCE_BASE` = D:\UNI\Sem6\Machine Learning\Project\dataset\dataset (workspace path [dataset/dataset](dataset/dataset)), `SECTIONS` = [sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec_a, sec_b, sec_c].
- [restructure_dataset.py](restructure_dataset.py): `SOURCE_BASE` = D:\UNI\Sem6\Machine Learning\Project\dataset\dataset (workspace path [dataset/dataset](dataset/dataset)), `DEST_BASE` = D:\UNI\Sem6\Machine Learning\Project\YOLODataset (workspace path [YOLODataset](YOLODataset)), `SECTIONS` = [sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec_a, sec_b, sec_c], `TRAIN_SPLIT` = 0.85.
- [Vehical detection/VehicleDetectionAndLabeling.py](Vehical%20detection/VehicleDetectionAndLabeling.py): `dataset_folder` = D:\UNI\Sem6\Machine Learning\Project\YOLODataset (workspace path [YOLODataset](YOLODataset)), `image_folder` = D:\UNI\Sem6\Machine Learning\Project\YOLODataset\val\images (workspace path [YOLODataset/val/images](YOLODataset/val/images)), `output_folder` = D:\UNI\Sem6\Machine Learning\Project\Results (workspace path [Results](Results)), `VEHICLE_CLASSES` = [car, motorcycle], `TRAIN_EPOCHS` = 50, `TRAIN_IMGSZ` = 1280, `TRAIN_BATCH` = 8, `TRAIN_DEVICE` = 0, `TRAIN_WORKERS` = 4, `TRAIN_PATIENCE` = 20, `USE_MIXED_PRECISION` = True, `CONFIDENCE_THRESHOLD` = 0.25, `IOU_THRESHOLD` = 0.45, `LOW_THRESHOLD` = 5, `MEDIUM_THRESHOLD` = 20.
- [Vehical detection/vehicledetectionwithfeatureextract.py](Vehical%20detection/vehicledetectionwithfeatureextract.py): `dataset_folder` = D:\UNI\Sem6\Machine Learning\Project\YOLODataset (workspace path [YOLODataset](YOLODataset)), `image_folder` = D:\UNI\Sem6\Machine Learning\Project\YOLODataset\val\images (workspace path [YOLODataset/val/images](YOLODataset/val/images)), `output_folder` = D:\UNI\Sem6\Machine Learning\Project\Results (workspace path [Results](Results)), `features_csv` = D:\UNI\Sem6\Machine Learning\Project\Results\ml_features.csv (workspace path [Results/ml_features.csv](Results/ml_features.csv)), `scaler_path` = D:\UNI\Sem6\Machine Learning\Project\Results\feature_scaler.pkl (workspace path [Results/feature_scaler.pkl](Results/feature_scaler.pkl)), `VEHICLE_CLASSES` = [car, motorcycle], `TRAIN_EPOCHS` = 50, `TRAIN_IMGSZ` = 1280, `TRAIN_BATCH` = 8, `TRAIN_DEVICE` = 0, `TRAIN_WORKERS` = 4, `TRAIN_PATIENCE` = 20, `USE_MIXED_PRECISION` = True, `CONFIDENCE_THRESHOLD` = 0.25, `IOU_THRESHOLD` = 0.45, `LOW_THRESHOLD` = 5, `MEDIUM_THRESHOLD` = 20.
- [Vehical detection/mlclassifiertraining.py](Vehical%20detection/mlclassifiertraining.py): `output_folder` = D:\UNI\Sem6\Machine Learning\Project\Results (workspace path [Results](Results)), `features_csv` = D:\UNI\Sem6\Machine Learning\Project\Results\ml_features.csv (workspace path [Results/ml_features.csv](Results/ml_features.csv)), `ml_ready_csv` = D:\UNI\Sem6\Machine Learning\Project\Results\ml_ready_dataset.csv (workspace path [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv)), `models_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\trained_models (workspace path [Results/trained_models](Results/trained_models)), `results_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\ml_results (workspace path [Results/ml_results](Results/ml_results)), and `select_best_features` is called with k = 15.
- [Vehical detection/statisticalanalysisandevaluation.py](Vehical%20detection/statisticalanalysisandevaluation.py): `output_folder` = D:\UNI\Sem6\Machine Learning\Project\Results (workspace path [Results](Results)), `models_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\trained_models (workspace path [Results/trained_models](Results/trained_models)), `results_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\ml_results (workspace path [Results/ml_results](Results/ml_results)), `evaluation_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\statistical_analysis (workspace path [Results/statistical_analysis](Results/statistical_analysis)), `ml_ready_csv` = D:\UNI\Sem6\Machine Learning\Project\Results\ml_ready_dataset.csv (workspace path [Results/ml_ready_dataset.csv](Results/ml_ready_dataset.csv)).
- [Vehical detection/advancedtrafficanalysiswithresearch.py](Vehical%20detection/advancedtrafficanalysiswithresearch.py): `output_folder` = D:\UNI\Sem6\Machine Learning\Project\Results (workspace path [Results](Results)), `models_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\trained_models (workspace path [Results/trained_models](Results/trained_models)), `advanced_results_folder` = D:\UNI\Sem6\Machine Learning\Project\Results\advanced_analysis (workspace path [Results/advanced_analysis](Results/advanced_analysis)).

## Results
The training phase produces per-model confusion matrices for 6 classifiers, feature-importance visualizations (Random Forest and other tree-based models), decision tree rules and visual diagrams, linear regression vehicle count predictions, a model comparison chart covering all 6 classifiers, and training summaries that report best accuracy, cross-validation scores, and top features. The linear regression model provides continuous vehicle count predictions that feed into density classification.

The statistical analysis phase expands this with dataset distribution plots, feature correlation and distribution analyses, ROC curves, a performance radar chart, and a consolidated statistical report with accuracy, precision, recall, F1-score metrics for each of the 6 classifiers, plus regression metrics (MSE, MAE, R²) for the linear regression model.

The advanced analysis phase creates [Results/advanced_analysis/research_integration_report.txt](Results/advanced_analysis/research_integration_report.txt) and [Results/advanced_analysis/research_integration_analysis.png](Results/advanced_analysis/research_integration_analysis.png) that summarize PoseRBPF and KalmanNet inspired features, along with [Results/advanced_analysis/advanced_features.csv](Results/advanced_analysis/advanced_features.csv) that augments the original feature set for further experimentation.

## Tech Stack
- Ultralytics YOLO for object detection training and inference.
- PyTorch for GPU training and the neural Kalman filter implementation.
- OpenCV for image loading and annotation.
- pandas and numpy for dataset and feature processing.
- PyYAML for writing YOLO data.yaml files.
- scikit-learn for classifiers, feature selection, metrics, and clustering.
- SciPy for convex hull based spatial feature calculations.
- matplotlib and seaborn for visualizations.
- joblib for model and sca ler persistence.

## Author
M. Shariq Shoaib , Safwan Ahmad Baseer , — University Project, Semester 6
