# statistical_analysis_and_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIGURATION
# -----------------------------
output_folder = r"D:\UNI\Sem6\Machine Learning\Project\Results"
models_folder = os.path.join(output_folder, "trained_models")
results_folder = os.path.join(output_folder, "ml_results")
evaluation_folder = os.path.join(output_folder, "statistical_analysis")
ml_ready_csv = os.path.join(output_folder, "ml_ready_dataset.csv")

# Create directories
os.makedirs(evaluation_folder, exist_ok=True)

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

# -----------------------------
# DATA LOADING AND PREPARATION
# -----------------------------
def load_data_and_models():
    """
    Load dataset and trained models
    """
    print("? LOADING DATA AND TRAINED MODELS...")
    
    # Load dataset
    if not os.path.exists(ml_ready_csv):
        raise FileNotFoundError("ML-ready dataset not found. Please run previous phases first.")
    
    df = pd.read_csv(ml_ready_csv)
    print(f"? Loaded dataset with {len(df)} samples")
    
    # Load models
    models = {}
    model_files = {
        'random_forest': 'random_forest_model.pkl',
        'svm': 'svm_model.pkl',
        'mlp': 'mlp_model.pkl',
        'knn': 'knn_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl',
        'decision_tree': 'decision_tree_model.pkl'
    }

    scalers = {}

    for model_name, model_file in model_files.items():
        model_path = os.path.join(models_folder, model_file)
        scaler_path = os.path.join(models_folder, f"{model_name}_scaler.pkl")

        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            print(f"? Loaded {model_name} model")

            if os.path.exists(scaler_path):
                scalers[model_name] = joblib.load(scaler_path)
                print(f"? Loaded {model_name} scaler")
        else:
            print(f"??  {model_name} model not found at {model_path}")
    
    if not models:
        raise FileNotFoundError("No trained models found. Please run Phase 7 first.")
    
    # Load feature selector
    selector_path = os.path.join(models_folder, "feature_selector.pkl")
    selected_features_path = os.path.join(models_folder, "selected_features.pkl")
    
    if os.path.exists(selector_path) and os.path.exists(selected_features_path):
        selector = joblib.load(selector_path)
        selected_features = joblib.load(selected_features_path)
        print(f"? Loaded feature selector with {len(selected_features)} features")
    else:
        print("??  Feature selector not found, using all features")
        selector = None
        selected_features = [col for col in df.columns if col not in ['image_name', 'traffic_label']]
    
    return df, models, scalers, selected_features, selector

# -----------------------------
# STATISTICAL GRAPHS - DATA DISTRIBUTION
# -----------------------------
def create_data_distribution_plots(df):
    """
    Create comprehensive data distribution plots
    """
    print("\n? CREATING DATA DISTRIBUTION PLOTS...")
    
    # 1. Class Distribution Pie Chart
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    class_counts = df['traffic_label'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Traffic Density Class Distribution', fontsize=14, fontweight='bold')
    
    # 2. Class Distribution Bar Chart
    plt.subplot(2, 3, 2)
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Traffic Density')
    plt.ylabel('Number of Images')
    
    # Add value labels on bars
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Vehicle Count Distribution
    plt.subplot(2, 3, 3)
    feature_columns = [col for col in df.columns if col not in ['image_name', 'traffic_label']]
    if 'vehicle_count' in df.columns:
        sns.histplot(data=df, x='vehicle_count', hue='traffic_label', multiple="stack", 
                    palette='viridis', bins=20)
        plt.title('Vehicle Count Distribution by Class', fontsize=14, fontweight='bold')
        plt.xlabel('Vehicle Count')
        plt.ylabel('Frequency')
    
    # 4. Feature Correlation Heatmap (Top 10 features)
    plt.subplot(2, 3, 4)
    numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        # Select top 10 most varying features for better visualization
        top_features = df[numeric_columns].std().sort_values(ascending=False).head(10).index
        correlation_matrix = df[top_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap (Top 10)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    # 5. Feature Distribution Box Plots
    plt.subplot(2, 3, 5)
    if len(numeric_columns) >= 3:
        # Select 3 important features for box plot
        important_features = ['vehicle_count', 'density_score', 'congestion_index']
        available_features = [f for f in important_features if f in df.columns][:3]
        
        if available_features:
            plot_data = df.melt(id_vars=['traffic_label'], value_vars=available_features,
                              var_name='Feature', value_name='Value')
            sns.boxplot(data=plot_data, x='Feature', y='Value', hue='traffic_label', palette='Set2')
            plt.title('Feature Distribution by Class', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Traffic Pattern Analysis
    plt.subplot(2, 3, 6)
    pattern_features = ['spatial_entropy', 'cluster_score', 'road_utilization']
    available_patterns = [f for f in pattern_features if f in df.columns]
    
    if available_patterns:
        pattern_data = df[['traffic_label'] + available_patterns].groupby('traffic_label').mean()
        pattern_data.plot(kind='bar', figsize=(10, 6), colormap='tab10')
        plt.title('Average Traffic Patterns by Class', fontsize=14, fontweight='bold')
        plt.xlabel('Traffic Density')
        plt.ylabel('Average Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    plt.tight_layout()
    distribution_plot_path = os.path.join(evaluation_folder, 'data_distribution_analysis.png')
    plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
    print(f"? Saved data distribution analysis: {distribution_plot_path}")
    plt.show()
    
    return class_counts

def create_feature_analysis_plots(df, selected_features):
    """
    Create detailed feature analysis plots
    """
    print("\n? CREATING DETAILED FEATURE ANALYSIS...")
    
    # 1. Feature Importance Comparison
    plt.figure(figsize=(15, 10))
    
    # Get top 10 features by variance
    numeric_df = df[selected_features].select_dtypes(include=[np.number])
    top_features = numeric_df.std().sort_values(ascending=False).head(10).index
    
    plt.subplot(2, 2, 1)
    feature_means = df[top_features].mean().sort_values(ascending=False)
    sns.barplot(x=feature_means.values, y=feature_means.index, palette='rocket')
    plt.title('Top 10 Features by Mean Value', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Value')
    
    plt.subplot(2, 2, 2)
    feature_stds = df[top_features].std().sort_values(ascending=False)
    sns.barplot(x=feature_stds.values, y=feature_stds.index, palette='viridis')
    plt.title('Top 10 Features by Standard Deviation', fontsize=14, fontweight='bold')
    plt.xlabel('Standard Deviation')
    
    # 3. Feature Distribution by Class
    plt.subplot(2, 2, 3)
    if 'vehicle_count' in df.columns:
        sns.violinplot(data=df, x='traffic_label', y='vehicle_count', palette='Set3')
        plt.title('Vehicle Count Distribution by Traffic Density', fontsize=14, fontweight='bold')
        plt.xlabel('Traffic Density')
        plt.ylabel('Vehicle Count')
    
    # 4. Correlation with Target
    plt.subplot(2, 2, 4)
    # Encode target for correlation calculation
    df_encoded = df.copy()
    df_encoded['traffic_label_encoded'] = df_encoded['traffic_label'].map(
        {'Low': 0, 'Medium': 1, 'High': 2}
    )
    
    correlations = {}
    for feature in top_features:
        if feature in df_encoded.columns:
            corr = df_encoded[feature].corr(df_encoded['traffic_label_encoded'])
            correlations[feature] = abs(corr)  # Use absolute correlation
    
    corr_series = pd.Series(correlations).sort_values(ascending=False)
    sns.barplot(x=corr_series.values, y=corr_series.index, palette='coolwarm')
    plt.title('Feature Correlation with Traffic Density', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Correlation Coefficient')
    
    plt.tight_layout()
    feature_analysis_path = os.path.join(evaluation_folder, 'feature_analysis.png')
    plt.savefig(feature_analysis_path, dpi=300, bbox_inches='tight')
    print(f"? Saved feature analysis: {feature_analysis_path}")
    plt.show()

# -----------------------------
# MODEL EVALUATION METRICS
# -----------------------------
def evaluate_all_models(df, models, scalers, selected_features, selector):
    """
    Comprehensive evaluation of all trained models
    """
    print("\n? COMPREHENSIVE MODEL EVALUATION...")
    
    # Prepare data
    X = df[selected_features]
    y = df['traffic_label']
    
    # Encode labels for multiclass ROC
    from sklearn.preprocessing import label_binarize
    y_encoded = label_binarize(y, classes=['Low', 'Medium', 'High'])
    n_classes = y_encoded.shape[1]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    evaluation_results = {}
    
    for model_name, model in models.items():
        print(f"\n? EVALUATING {model_name.upper()}...")
        
        # Prepare data for specific model
        if model_name in scalers:
            scaler = scalers[model_name]
            X_test_scaled = scaler.transform(X_test)
            X_test_processed = X_test_scaled
        else:
            X_test_processed = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed) if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store results
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        print(f"   ? Accuracy: {accuracy:.4f}")
        print(f"   ? Precision: {precision:.4f}")
        print(f"   ? Recall: {recall:.4f}")
        print(f"   ? F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print(f"\n   ? Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    
    return evaluation_results, X_test, y_test

# -----------------------------
# COMPREHENSIVE VISUALIZATIONS
# -----------------------------
def create_model_comparison_visualizations(evaluation_results):
    """
    Create comprehensive model comparison visualizations
    """
    print("\n? CREATING MODEL COMPARISON VISUALIZATIONS...")
    
    # 1. Performance Metrics Comparison
    metrics_df = pd.DataFrame({
        model: {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'], 
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        }
        for model, results in evaluation_results.items()
    }).T
    
    plt.figure(figsize=(15, 12))
    
    # Metrics Comparison Bar Chart
    plt.subplot(2, 2, 1)
    metrics_df.plot(kind='bar', figsize=(12, 8), colormap='Set2', edgecolor='black')
    plt.title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Machine Learning Models', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, model in enumerate(metrics_df.index):
        for j, metric in enumerate(metrics_df.columns):
            plt.text(i + j*0.2 - 0.3, metrics_df.loc[model, metric] + 0.01, 
                    f'{metrics_df.loc[model, metric]:.3f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. Confusion Matrices Grid
    plt.subplot(2, 2, 2)
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        axes[idx].set_title(f'{model_name.title()}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    
    # 3. ROC Curves (for models with probability estimates)
    plt.subplot(2, 2, 3)
    from sklearn.preprocessing import label_binarize
    
    models_with_proba = {name: results for name, results in evaluation_results.items() 
                        if results['probabilities'] is not None}
    
    if models_with_proba:
        y_test_combined = list(models_with_proba.values())[0]['true_labels']
        y_test_bin = label_binarize(y_test_combined, classes=['Low', 'Medium', 'High'])
        
        for model_name, results in models_with_proba.items():
            y_score = results['probabilities']
            
            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"],
                    label=f'{model_name} (AUC = {roc_auc["micro"]:0.2f})',
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Micro-average ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    # 4. Performance Radar Chart
    plt.subplot(2, 2, 4)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    for model_name, results in evaluation_results.items():
        values = [
            results['accuracy'],
            results['precision'], 
            results['recall'],
            results['f1_score']
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name.title())
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    comparison_path = os.path.join(evaluation_folder, 'comprehensive_model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"? Saved comprehensive model comparison: {comparison_path}")
    plt.show()
    
    return metrics_df

def create_detailed_performance_analysis(evaluation_results):
    """
    Create detailed performance analysis reports and visualizations
    """
    print("\n? CREATING DETAILED PERFORMANCE ANALYSIS...")
    
    # Create performance summary table
    performance_data = []
    for model_name, results in evaluation_results.items():
        performance_data.append({
            'Model': model_name.title(),
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}", 
            'F1-Score': f"{results['f1_score']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Save performance table
    performance_csv_path = os.path.join(evaluation_folder, 'model_performance_metrics.csv')
    performance_df.to_csv(performance_csv_path, index=False)
    print(f"? Saved performance metrics: {performance_csv_path}")
    
    # Create best model analysis
    best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
    best_model_name, best_model_results = best_model
    
    print(f"\n? BEST PERFORMING MODEL: {best_model_name.upper()}")
    print(f"   Accuracy: {best_model_results['accuracy']:.4f}")
    print(f"   Precision: {best_model_results['precision']:.4f}")
    print(f"   Recall: {best_model_results['recall']:.4f}")
    print(f"   F1-Score: {best_model_results['f1_score']:.4f}")
    
    # Create detailed confusion matrix for best model
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(best_model_results['true_labels'], best_model_results['predictions'])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {best_model_name.title()}\n(Accuracy: {best_model_results["accuracy"]:.4f})', 
              fontsize=16, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Low', 'Medium', 'High'], rotation=45)
    plt.yticks(tick_marks, ['Low', 'Medium', 'High'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight='bold')
    
    plt.tight_layout()
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    best_cm_path = os.path.join(evaluation_folder, 'best_model_confusion_matrix.png')
    plt.savefig(best_cm_path, dpi=300, bbox_inches='tight')
    print(f"? Saved best model confusion matrix: {best_cm_path}")
    plt.show()
    
    return performance_df, best_model_name

# -----------------------------
# STATISTICAL REPORT GENERATION
# -----------------------------
def generate_statistical_report(df, evaluation_results, performance_df, best_model_name, class_counts):
    """
    Generate comprehensive statistical report with regression metrics
    """
    print("\n? GENERATING STATISTICAL REPORT...")

    # Load regression model if available
    linreg_info = ""
    linreg_path = os.path.join(models_folder, 'linear_regression_model.pkl')
    if os.path.exists(linreg_path):
        try:
            linreg_model = joblib.load(linreg_path)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.model_selection import train_test_split

            # Prepare data for regression evaluation
            ml_ready_csv_path = os.path.join(output_folder, "ml_ready_dataset.csv")
            if os.path.exists(ml_ready_csv_path):
                df_reg = pd.read_csv(ml_ready_csv_path)
                if 'vehicle_count' in df_reg.columns:
                    feature_columns = [col for col in df_reg.columns if col not in ['image_name', 'traffic_label', 'vehicle_count']]
                    X_reg = df_reg[feature_columns]
                    y_reg = df_reg['vehicle_count']

                    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                        X_reg, y_reg, test_size=0.2, random_state=42
                    )

                    y_pred_reg = linreg_model.predict(X_test_r)
                    mse_reg = mean_squared_error(y_test_r, y_pred_reg)
                    mae_reg = mean_absolute_error(y_test_r, y_pred_reg)
                    r2_reg = r2_score(y_test_r, y_pred_reg)

                    linreg_info = f"""
LINEAR REGRESSION MODEL (Vehicle Count Prediction)
---------------------------------------------------
Mean Squared Error: {mse_reg:.4f}
Mean Absolute Error: {mae_reg:.4f}
R? Score: {r2_reg:.4f}

"""
        except Exception as e:
            print(f"??  Could not evaluate linear regression: {e}")

    report = f"""
SKYTRAFFIC AI - STATISTICAL ANALYSIS REPORT
===========================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
----------------
Total Samples: {len(df):,}
Number of Features: {len([col for col in df.columns if col not in ['image_name', 'traffic_label']])}
Class Distribution:
  - Low Traffic: {class_counts.get('Low', 0):,} samples ({class_counts.get('Low', 0)/len(df)*100:.1f}%)
  - Medium Traffic: {class_counts.get('Medium', 0):,} samples ({class_counts.get('Medium', 0)/len(df)*100:.1f}%)
  - High Traffic: {class_counts.get('High', 0):,} samples ({class_counts.get('High', 0)/len(df)*100:.1f}%)

CLASSIFIER PERFORMANCE SUMMARY
------------------------------
Best Performing Model: {best_model_name.title()}

Detailed Performance Metrics:
{performance_df.to_string(index=False)}

{linreg_info}
KEY FINDINGS
------------
"""

    # Add key findings based on performance
    best_accuracy = max([results['accuracy'] for results in evaluation_results.values()])

    if best_accuracy >= 0.9:
        report += "*** EXCELLENT PERFORMANCE: Models achieve over 90% accuracy\n"
    elif best_accuracy >= 0.8:
        report += "*** GOOD PERFORMANCE: Models achieve 80-90% accuracy\n"
    elif best_accuracy >= 0.7:
        report += "*** MODERATE PERFORMANCE: Models achieve 70-80% accuracy\n"
    else:
        report += "*** POOR PERFORMANCE: Models below 70% accuracy - consider feature engineering\n"

    # Add class-wise performance insights
    report += f"\nCLASSIFICATION INSIGHTS:\n"
    report += f"- Best model accuracy: {best_accuracy:.1%}\n"
    report += f"- Number of classifiers evaluated: {len(evaluation_results)}\n"

    # Check for class imbalance issues
    min_class = min(class_counts.values)
    max_class = max(class_counts.values)
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

    if imbalance_ratio > 2:
        report += f"- *** Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)\n"
    else:
        report += f"- *** Balanced dataset (ratio: {imbalance_ratio:.1f}:1)\n"

    # Add model-specific insights
    report += f"\nMODEL-SPECIFIC INSIGHTS:\n"
    for model_name, results in evaluation_results.items():
        report += f"- {model_name.title()}: Accuracy = {results['accuracy']:.3f}, F1-Score = {results['f1_score']:.3f}\n"

    # Add recommendations
    report += f"\nRECOMMENDATIONS:\n"
    if best_accuracy >= 0.85:
        report += "- Model performance is excellent for deployment\n"
        report += "- Consider collecting more diverse data for generalization\n"
    elif best_accuracy >= 0.75:
        report += "- Model performance is good but can be improved\n"
        report += "- Consider hyperparameter tuning and feature engineering\n"
    else:
        report += "- Model performance needs significant improvement\n"
        report += "- Consider collecting more data and trying different algorithms\n"

    # Save report with UTF-8 encoding to handle any special characters
    report_path = os.path.join(evaluation_folder, 'statistical_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"? Saved statistical report: {report_path}")

    # Print report to console
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS REPORT SUMMARY")
    print("="*60)
    print(report)

    return report
# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    print("? PHASE 8 ? STATISTICAL GRAPHS & MODEL EVALUATION")
    print("=" * 60)
    print("? Creating comprehensive statistical analysis")
    print("? Evaluating 6 classifiers + 1 regression model")
    print("? Producing evaluation reports")
    print("=" * 60)
    
    try:
        # 1. Load data and models
        df, models, scalers, selected_features, selector = load_data_and_models()
        
        # 2. Create data distribution plots
        class_counts = create_data_distribution_plots(df)
        
        # 3. Create feature analysis plots
        create_feature_analysis_plots(df, selected_features)
        
        # 4. Evaluate all models
        evaluation_results, X_test, y_test = evaluate_all_models(
            df, models, scalers, selected_features, selector
        )
        
        # 5. Create model comparison visualizations
        metrics_df = create_model_comparison_visualizations(evaluation_results)
        
        # 6. Create detailed performance analysis
        performance_df, best_model_name = create_detailed_performance_analysis(evaluation_results)
        
        # 7. Generate statistical report
        report = generate_statistical_report(df, evaluation_results, performance_df, best_model_name, class_counts)
        
        print(f"\n? PHASE 8 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("? OUTPUT FILES GENERATED:")
        print(f"   ? Data distribution analysis: {evaluation_folder}/data_distribution_analysis.png")
        print(f"   ? Feature analysis: {evaluation_folder}/feature_analysis.png")
        print(f"   ? Model comparisons: {evaluation_folder}/comprehensive_model_comparison.png")
        print(f"   ? Performance metrics: {evaluation_folder}/model_performance_metrics.csv")
        print(f"   ? Statistical report: {evaluation_folder}/statistical_analysis_report.txt")
        print(f"   ? Best model analysis: {evaluation_folder}/best_model_confusion_matrix.png")
        print(f"\n? KEY FINDINGS:")
        best_accuracy = max([results['accuracy'] for results in evaluation_results.values()])
        print(f"   Best Model Accuracy: {best_accuracy:.4f}")
        print(f"   Total Visualizations: 6+ comprehensive charts")
        print(f"   Statistical Analysis: Complete")
        
    except Exception as e:
        print(f"? Error in statistical analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
