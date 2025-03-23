import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load preprocessed dataset
df = pd.read_csv("data/processed/parkinsons_preprocessed.csv")

# Feature columns
feature_columns = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", 
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
    "FamilyHistoryParkinsons", "TraumaticBrainInjury", "Hypertension", "Diabetes", 
    "Depression", "Stroke", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "UPDRS", 
    "MoCA", "FunctionalAssessment", "Tremor", "Rigidity", "Bradykinesia", 
    "PosturalInstability", "SpeechProblems", "SleepDisorders", "Constipation"
]

# Ensure dataset contains all required features
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print("\033[91m⚠️ WARNING: Missing features in dataset:\033[0m", missing_features)
    exit()

# Define features (X) and target (y)
X = df[feature_columns]
y = df["Diagnosis"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predictions
rf_preds = rf_model.predict(X_test)
knn_preds = knn_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_preds)
knn_accuracy = accuracy_score(y_test, knn_preds)

# Styled Output
print("\n\033[1m🔍 Model Performance Summary\033[0m")
print("-" * 50)
print(f"\033[1;34m📊 Random Forest Accuracy:\033[0m {rf_accuracy:.4f} " + "█" * int(rf_accuracy * 20))
print(f"\033[1;33m📊 KNN Accuracy:\033[0m {knn_accuracy:.4f} " + "█" * int(knn_accuracy * 20))
print("-" * 50)

# Display Classification Reports
print("\n\033[1;36m📝 Random Forest Classification Report:\033[0m")
print(classification_report(y_test, rf_preds))

print("\n\033[1;35m📝 KNN Classification Report:\033[0m")
print(classification_report(y_test, knn_preds))

# Confusion Matrix Function
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Parkinson's", "Parkinson's"], yticklabels=["No Parkinson's", "Parkinson's"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Plot Confusion Matrices
plot_confusion_matrix(y_test, rf_preds, "Random Forest")
plot_confusion_matrix(y_test, knn_preds, "KNN")

# Save models
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(knn_model, "models/knn.pkl")

print("\n✅ \033[92mModels saved successfully!\033[0m")
