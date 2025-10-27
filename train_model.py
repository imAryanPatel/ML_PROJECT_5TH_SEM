import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_csv('data/Crop_recommendation.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Drop unnamed columns if they exist
columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)
    print(f"\nDropped columns: {columns_to_drop}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Handle outliers
numerical_cols = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']

print("\nHandling outliers...")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print("Outliers handled successfully!")

# Prepare features and target
X = df.drop('label', axis=1)
y = df['label']

print(f"\nUnique crops: {y.nunique()}")
print(f"Crops: {y.unique()}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoding labels...")
print(f"Label encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaling features...")
print(f"Feature means after scaling: {X_scaled.mean(axis=0)}")
print(f"Feature stds after scaling: {X_scaled.std(axis=0)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train base Random Forest model
print("\nTraining base Random Forest model...")
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)

base_train_acc = rf_base.score(X_train, y_train)
base_test_acc = rf_base.score(X_test, y_test)

print(f"Base model - Train Accuracy: {base_train_acc:.4f}")
print(f"Base model - Test Accuracy: {base_test_acc:.4f}")

# Hyperparameter tuning
print("\nPerforming hyperparameter tuning...")
print("This may take a few minutes...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\nBest parameters: {best_params}")

# Evaluate best model
train_accuracy = best_rf_model.score(X_train, y_train)
test_accuracy = best_rf_model.score(X_test, y_test)

print(f"\nBest model - Train Accuracy: {train_accuracy:.4f}")
print(f"Best model - Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = best_rf_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save models
print("\nSaving models...")

with open('models/best_random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)
print("✓ Saved: models/best_random_forest_model.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: models/scaler.pkl")

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✓ Saved: models/label_encoder.pkl")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print("\nYou can now run the Streamlit app:")
print("streamlit run app.py")