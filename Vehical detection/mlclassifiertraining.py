# ml_classifier_training_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIGURATION
# -----------------------------
output_folder = r"D:\UNI\Sem6\Machine Learning\Project\Results"
features_csv = os.path.join(output_folder, "ml_features.csv")
ml_ready_csv = os.path.join(output_folder, "ml_ready_dataset.csv")
models_folder = os.path.join(output_folder, "trained_models")
results_folder = os.path.join(output_folder, "ml_results")

# Create directories
os.makedirs(models_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# -----------------------------
# DATA LOADING AND PREPARATION
# -----------------------------
def load_and_prepare_data():
    """
    Load the extracted features and prepare for ML training
    """
    print("📊 LOADING AND PREPARING DATA FOR ML TRAINING...")
    
    # Try to load ML-ready dataset first
    if os.path.exists(ml_ready_csv):
        df = pd.read_csv(ml_ready_csv)
        print("✅ Loaded ML-ready dataset")
    elif os.path.exists(features_csv):
        df = pd.read_csv(features_csv)
        print("✅ Loaded raw features dataset")
        
        # If raw features, we need to prepare them for ML
        feature_columns = [col for col in df.columns if col not in ['image_name', 'traffic_label']]
        
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Save as ML-ready
        df.to_csv(ml_ready_csv, index=False)
        joblib.dump(scaler, os.path.join(output_folder, "feature_scaler.pkl"))
        print("✅ Created ML-ready dataset from raw features")
    else:
        raise FileNotFoundError("No feature dataset found. Please run feature extraction first.")
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['image_name', 'traffic_label']]
    X = df[feature_columns]
    y = df['traffic_label']
    
    print(f"🎯 Features: {len(feature_columns)}")
    print(f"🎯 Target classes: {y.unique()}")
    print(f"🎯 Class distribution:\n{y.value_counts()}")
    
    return X, y, feature_columns, df

def explore_features(X, y, feature_columns):
    """
    Perform exploratory data analysis on features
    """
    print("\n🔍 EXPLORATORY DATA ANALYSIS...")
    
    # Create feature importance plot using Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance', fontweight='bold')
    plt.ylabel('Features', fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    feature_plot_path = os.path.join(results_folder, 'feature_importance.png')
    plt.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved feature importance plot: {feature_plot_path}")
    plt.show()
    
    print("📊 Top 10 Most Important Features:")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance_df

def select_best_features(X, y, feature_columns, k=15):
    """
    Select best features using statistical tests
    """
    print(f"\n🎯 SELECTING BEST {k} FEATURES...")
    
    # Use SelectKBest for feature selection
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selected_mask[i]]
    
    print(f"✅ Selected {len(selected_features)} features:")
    for feature in selected_features:
        print(f"   - {feature}")
    
    return X_selected, selected_features, selector

# -----------------------------
# MODEL TRAINING FUNCTIONS
# -----------------------------
def train_random_forest(X_train, X_test, y_train, y_test, selected_features):
    """
    Train and evaluate Random Forest classifier
    """
    print("\n🌲 TRAINING RANDOM FOREST CLASSIFIER...")
    
    # Simplified hyperparameter tuning for faster execution
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    
    print(f"✅ Random Forest Results:")
    print(f"   Best Parameters: {grid_search.best_params_}")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_rf, y_pred, y_pred_proba, accuracy, cv_scores.mean()

def train_svm(X_train, X_test, y_train, y_test, selected_features):
    """
    Train and evaluate Support Vector Machine classifier
    """
    print("\n⚡ TRAINING SUPPORT VECTOR MACHINE...")
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simplified hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.1],
        'kernel': ['rbf']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)
    y_pred_proba = best_svm.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
    
    print(f"✅ SVM Results:")
    print(f"   Best Parameters: {grid_search.best_params_}")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_svm, y_pred, y_pred_proba, accuracy, cv_scores.mean(), scaler

def train_mlp(X_train, X_test, y_train, y_test, selected_features):
    """
    Train and evaluate Multi-Layer Perceptron classifier
    """
    print("\n🧠 TRAINING MULTI-LAYER PERCEPTRON...")
    
    # Scale features for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simplified hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'alpha': [0.001, 0.01]
    }
    
    mlp = MLPClassifier(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_mlp = grid_search.best_estimator_
    y_pred = best_mlp.predict(X_test_scaled)
    y_pred_proba = best_mlp.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_mlp, X_train_scaled, y_train, cv=5)
    
    print(f"✅ MLP Results:")
    print(f"   Best Parameters: {grid_search.best_params_}")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_mlp, y_pred, y_pred_proba, accuracy, cv_scores.mean(), scaler

# -----------------------------
# MODEL EVALUATION AND VISUALIZATION
# -----------------------------
def evaluate_model(model, model_name, X_test, y_test, y_pred, y_pred_proba, class_names, selected_features):
    """
    Comprehensive model evaluation with visualizations
    """
    print(f"\n📊 EVALUATING {model_name.upper()} MODEL...")
    
    # Classification report
    print(f"📈 Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = os.path.join(results_folder, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix: {cm_path}")
    plt.show()
    
    # Feature importance (for Random Forest)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
        plt.title(f'Top 10 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Importance', fontweight='bold')
        plt.ylabel('Features', fontweight='bold')
        plt.tight_layout()
        
        # Save feature importance
        fi_path = os.path.join(results_folder, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved feature importance: {fi_path}")
        plt.show()

def create_model_comparison_chart(model_results):
    """
    Create and save model comparison chart
    """
    print("\n📊 CREATING MODEL COMPARISON CHART...")
    
    comparison_df = pd.DataFrame(model_results)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    models = comparison_df['model']
    accuracies = comparison_df['accuracy']
    cv_scores = comparison_df['cv_score']
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x + width/2, cv_scores, width, label='CV Score', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
    plt.ylabel('Performance Scores', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, v in zip(bars1, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    for bar, v in zip(bars2, cv_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save comparison chart
    comparison_path = os.path.join(results_folder, 'model_comparison_chart.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved model comparison chart: {comparison_path}")
    plt.show()
    
    return comparison_df

def create_training_summary(comparison_df, feature_importance_df, selected_features, df, y):
    """
    Create comprehensive training summary with proper type conversion for JSON
    """
    print("\n📝 CREATING TRAINING SUMMARY...")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Create summary dictionary with serializable types
    summary = {
        'training_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_model': str(comparison_df.iloc[0]['model']),  # Ensure string
        'best_accuracy': float(comparison_df.iloc[0]['accuracy']),
        'best_cv_score': float(comparison_df.iloc[0]['cv_score']),
        'total_samples': int(len(df)),
        'feature_count': int(len(selected_features)),
        'class_distribution': {str(k): int(v) for k, v in dict(y.value_counts()).items()},
        'top_5_features': [
            {
                'feature': str(row['feature']),
                'importance': float(row['importance'])
            }
            for _, row in feature_importance_df.head(5).iterrows()
        ],
        'model_performance': [
            {
                'model': str(row['model']),
                'accuracy': float(row['accuracy']),
                'cv_score': float(row['cv_score'])
            }
            for _, row in comparison_df.iterrows()
        ]
    }
    
    # Apply serialization conversion to entire summary
    summary = convert_to_serializable(summary)
    
    # Save summary as JSON
    import json
    summary_path = os.path.join(results_folder, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Saved training summary: {summary_path}")
    
    # Create readable text summary
    text_summary = f"""
TRAINING SUMMARY REPORT
=======================
Generated: {summary['training_timestamp']}

DATASET INFORMATION
-------------------
Total Samples: {summary['total_samples']}
Features Used: {summary['feature_count']}
Class Distribution: {summary['class_distribution']}

MODEL PERFORMANCE
-----------------
Best Model: {summary['best_model']}
Best Accuracy: {summary['best_accuracy']:.4f}
Best CV Score: {summary['best_cv_score']:.4f}

TOP 5 FEATURES
--------------
"""
    for i, feat in enumerate(summary['top_5_features'], 1):
        text_summary += f"{i}. {feat['feature']}: {feat['importance']:.4f}\n"
    
    text_summary += "\nDETAILED MODEL PERFORMANCE\n-------------------------\n"
    for model in summary['model_performance']:
        text_summary += f"{model['model']}:\n"
        text_summary += f"  Accuracy: {model['accuracy']:.4f}\n"
        text_summary += f"  CV Score: {model['cv_score']:.4f}\n"
    
    # Save text summary
    text_summary_path = os.path.join(results_folder, 'training_summary.txt')
    with open(text_summary_path, 'w') as f:
        f.write(text_summary)
    print(f"✅ Saved text summary: {text_summary_path}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(text_summary)
    
    return summary

def save_models(trained_models):
    """
    Save all trained models and metadata
    """
    print("\n💾 SAVING TRAINED MODELS...")
    
    for model_info in trained_models:
        model_name = model_info['name']
        model = model_info['model']
        accuracy = model_info['accuracy']
        
        # Save model
        model_path = os.path.join(models_folder, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)
        
        # Save scaler if exists
        if 'scaler' in model_info and model_info['scaler'] is not None:
            scaler_path = os.path.join(models_folder, f'{model_name}_scaler.pkl')
            joblib.dump(model_info['scaler'], scaler_path)
        
        print(f"✅ Saved {model_name}: {model_path} (Accuracy: {accuracy:.4f})")
    
    print(f"📁 All models saved in: {models_folder}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    print("🟥 PHASE 7 — SUPERVISED ML CLASSIFIER TRAINING")
    print("=" * 60)
    print("🎯 Training Random Forest, SVM, and MLP classifiers")
    print("🎯 Target: Traffic Density (Low/Medium/High)")
    print("=" * 60)
    
    try:
        # 1. Load and prepare data
        X, y, feature_columns, df = load_and_prepare_data()
        
        # 2. Explore features
        feature_importance_df = explore_features(X, y, feature_columns)
        
        # 3. Select best features
        X_selected, selected_features, selector = select_best_features(X, y, feature_columns, k=15)
        
        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 DATA SPLIT:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Features: {X_train.shape[1]}")
        
        # 5. Train models
        trained_models = []
        model_results = []
        
        # Random Forest
        rf_model, rf_pred, rf_proba, rf_accuracy, rf_cv_score = train_random_forest(
            X_train, X_test, y_train, y_test, selected_features
        )
        trained_models.append({
            'name': 'random_forest',
            'model': rf_model,
            'accuracy': rf_accuracy,
            'scaler': None
        })
        model_results.append({
            'model': 'Random Forest',
            'accuracy': rf_accuracy,
            'cv_score': rf_cv_score
        })
        evaluate_model(rf_model, 'Random Forest', X_test, y_test, rf_pred, rf_proba, y.unique(), selected_features)
        
        # SVM
        svm_model, svm_pred, svm_proba, svm_accuracy, svm_cv_score, svm_scaler = train_svm(
            X_train, X_test, y_train, y_test, selected_features
        )
        trained_models.append({
            'name': 'svm',
            'model': svm_model,
            'accuracy': svm_accuracy,
            'scaler': svm_scaler
        })
        model_results.append({
            'model': 'SVM',
            'accuracy': svm_accuracy,
            'cv_score': svm_cv_score
        })
        evaluate_model(svm_model, 'SVM', X_test, y_test, svm_pred, svm_proba, y.unique(), selected_features)
        
        # MLP
        mlp_model, mlp_pred, mlp_proba, mlp_accuracy, mlp_cv_score, mlp_scaler = train_mlp(
            X_train, X_test, y_train, y_test, selected_features
        )
        trained_models.append({
            'name': 'mlp',
            'model': mlp_model,
            'accuracy': mlp_accuracy,
            'scaler': mlp_scaler
        })
        model_results.append({
            'model': 'MLP',
            'accuracy': mlp_accuracy,
            'cv_score': mlp_cv_score
        })
        evaluate_model(mlp_model, 'MLP', X_test, y_test, mlp_pred, mlp_proba, y.unique(), selected_features)
        
        # 6. Create comparison chart
        comparison_df = create_model_comparison_chart(model_results)
        
        # 7. Create training summary
        summary = create_training_summary(comparison_df, feature_importance_df, selected_features, df, y)
        
        # 8. Save models
        save_models(trained_models)
        
        # 9. Save feature selector and metadata
        joblib.dump(selector, os.path.join(models_folder, 'feature_selector.pkl'))
        joblib.dump(selected_features, os.path.join(models_folder, 'selected_features.pkl'))
        
        print(f"\n🎉 PHASE 7 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 FINAL RESULTS SUMMARY:")
        print(f"   Best Model: {summary['best_model']}")
        print(f"   Best Accuracy: {summary['best_accuracy']:.4f}")
        print(f"   Features Used: {summary['feature_count']}")
        print(f"   Total Samples: {summary['total_samples']}")
        print(f"   Class Distribution: {summary['class_distribution']}")
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   📊 Model comparison chart: {results_folder}/model_comparison_chart.png")
        print(f"   📝 Training summary: {results_folder}/training_summary.json")
        print(f"   📝 Text summary: {results_folder}/training_summary.txt")
        print(f"   🎯 Feature importance plots")
        print(f"   📈 Confusion matrices")
        print(f"   🤖 Trained models: {models_folder}/")
        
    except Exception as e:
        print(f"❌ Error in ML training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
