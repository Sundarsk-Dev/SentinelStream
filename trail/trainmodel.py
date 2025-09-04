# Credit Card Fraud Detection Model
# Run this in VS Code with Python extension installed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_auc_score, 
                           average_precision_score, roc_curve)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Step 1: Download dataset using kagglehub
print("Step 1: Downloading dataset...")
try:
    import kagglehub
    # Download latest version
    path = kagglehub.dataset_download("whenamancodes/fraud-detection")
    print("Path to dataset files:", path)
    
    # Load the dataset
    # The dataset is usually named creditcard.csv
    import os
    files = os.listdir(path)
    print("Available files:", files)
    
    # Find the CSV file
    csv_file = None
    for file in files:
        if file.endswith('.csv'):
            csv_file = file
            break
    
    if csv_file:
        data_path = os.path.join(path, csv_file)
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully from: {data_path}")
    else:
        print("CSV file not found. Please check the downloaded files.")
        
except ImportError:
    print("kagglehub not installed. Install it using: pip install kagglehub")
    print("For now, loading sample data structure...")
    # Create sample data structure for demonstration
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame()
    
    # Create PCA features V1-V28
    for i in range(1, 29):
        df[f'V{i}'] = np.random.randn(n_samples)
    
    df['Time'] = np.random.randint(0, 172800, n_samples)  # 2 days in seconds
    df['Amount'] = np.random.lognormal(3, 1.5, n_samples)
    df['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])  # Imbalanced
    
    print("Sample dataset created for demonstration")

print("\nStep 2: Exploring the dataset...")
print(f"Dataset shape: {df.shape}")
print(f"Dataset info:")
print(df.info())

# Display basic statistics
print("\nDataset description:")
print(df.describe())

# Display input features for pipeline
print("\nStep 3: Input Features for Pipeline:")
print("="*50)
feature_columns = [col for col in df.columns if col != 'Class']
print(f"Total number of input features: {len(feature_columns)}")
print("\nFeature categories:")
print("1. PCA-transformed features (V1-V28):")
pca_features = [col for col in feature_columns if col.startswith('V')]
print(f"   {pca_features}")
print(f"   Count: {len(pca_features)}")

print("\n2. Original features (not PCA-transformed):")
original_features = [col for col in feature_columns if not col.startswith('V')]
print(f"   {original_features}")
print(f"   Count: {len(original_features)}")

print(f"\n3. Target variable: 'Class' (0 = Normal, 1 = Fraud)")

# Class distribution
print("\nStep 4: Class Distribution Analysis:")
class_dist = df['Class'].value_counts()
print(f"Class distribution:")
print(f"Normal transactions (Class 0): {class_dist[0]:,} ({class_dist[0]/len(df)*100:.3f}%)")
print(f"Fraudulent transactions (Class 1): {class_dist[1]:,} ({class_dist[1]/len(df)*100:.3f}%)")

# Visualizations
plt.figure(figsize=(15, 10))

# Class distribution
plt.subplot(2, 3, 1)
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

# Amount distribution by class
plt.subplot(2, 3, 2)
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, label='Normal', density=True)
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, label='Fraud', density=True)
plt.xlabel('Amount')
plt.ylabel('Density')
plt.legend()
plt.title('Transaction Amount Distribution')
plt.xlim(0, 1000)  # Limit for better visualization

# Time distribution
plt.subplot(2, 3, 3)
df['Time'].hist(bins=50)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.title('Transaction Time Distribution')

# Correlation heatmap of a subset of features
plt.subplot(2, 3, 4)
corr_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Time', 'Amount', 'Class']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')

# Amount vs Class boxplot
plt.subplot(2, 3, 5)
df.boxplot(column='Amount', by='Class')
plt.title('Amount Distribution by Class')
plt.suptitle('')

# Feature importance preview (using a simple model)
plt.subplot(2, 3, 6)
from sklearn.ensemble import RandomForestClassifier
X_sample = df[feature_columns].sample(min(10000, len(df)))
y_sample = df.loc[X_sample.index, 'Class']
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_sample, y_sample)
feature_importance = pd.Series(rf_temp.feature_importances_, index=feature_columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

print("\nStep 5: Data Preprocessing...")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set fraud ratio: {y_train.sum()/len(y_train)*100:.3f}%")

# Feature scaling (important for logistic regression and other algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStep 6: Handling Class Imbalance...")

# Method 1: SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Training samples: {X_train_smote.shape[0]}")
print(f"After SMOTE - Fraud ratio: {y_train_smote.sum()/len(y_train_smote)*100:.1f}%")

print("\nStep 7: Model Training...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        # Use SMOTE data for logistic regression
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Use original data for Random Forest (with class_weight for balanced version)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    results[name] = {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }
    
    print(f"{name} Results:")
    print(f"ROC AUC: {auc_roc:.4f}")
    print(f"PR AUC (recommended): {auc_pr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\nStep 8: Model Evaluation and Comparison...")

# Plot ROC and PR curves
plt.figure(figsize=(15, 5))

# ROC Curve
plt.subplot(1, 3, 1)
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc_roc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 3, 2)
for name, result in results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
    plt.plot(recall, precision, label=f"{name} (AUC = {result['auc_pr']:.3f})")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()

# Model comparison
plt.subplot(1, 3, 3)
model_names = list(results.keys())
auc_roc_scores = [results[name]['auc_roc'] for name in model_names]
auc_pr_scores = [results[name]['auc_pr'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, auc_roc_scores, width, label='ROC AUC', alpha=0.8)
plt.bar(x + width/2, auc_pr_scores, width, label='PR AUC', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('AUC Score')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Select best model based on PR AUC (recommended for imbalanced data)
best_model_name = max(results.keys(), key=lambda x: results[x]['auc_pr'])
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best PR AUC Score: {results[best_model_name]['auc_pr']:.4f}")

# Detailed confusion matrix for best model
print(f"\nDetailed Evaluation for {best_model_name}:")
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
print("Confusion Matrix:")
print(cm)

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nDetailed Metrics:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
print(f"Precision: {tp/(tp+fp):.4f}")
print(f"Recall (Sensitivity): {tp/(tp+fn):.4f}")
print(f"Specificity: {tn/(tn+fp):.4f}")

print("\nStep 9: Feature Importance Analysis...")
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importance.head(15).plot(x='feature', y='importance', kind='barh')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

print("\nStep 10: Pipeline Summary for Production...")
print("="*60)
print("COMPLETE MACHINE LEARNING PIPELINE SUMMARY")
print("="*60)
print(f"1. Input Features: {len(feature_columns)} features")
print(f"   - PCA Features: {pca_features}")
print(f"   - Original Features: {original_features}")
print(f"2. Preprocessing: StandardScaler normalization")
print(f"3. Class Imbalance Handling: SMOTE oversampling")
print(f"4. Best Model: {best_model_name}")
print(f"5. Performance Metrics:")
print(f"   - PR AUC (Primary): {results[best_model_name]['auc_pr']:.4f}")
print(f"   - ROC AUC: {results[best_model_name]['auc_roc']:.4f}")
print(f"6. Recommendation: Use PR AUC for model evaluation due to class imbalance")

# Save the model and scaler for future use
import joblib
joblib.dump(best_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'fraud_detection_scaler.pkl')
print(f"\n7. Model and scaler saved as:")
print(f"   - fraud_detection_model.pkl")
print(f"   - fraud_detection_scaler.pkl")

print("\nTraining completed successfully!")
print("You can now use this model to predict fraud in new transactions.")