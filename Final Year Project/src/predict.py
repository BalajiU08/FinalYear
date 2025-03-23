import os
import pandas as pd
import joblib

# Define feature names
feature_names = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", 
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", 
    "FamilyHistoryParkinsons", "TraumaticBrainInjury", "Hypertension", "Diabetes", 
    "Depression", "Stroke", "SystolicBP", "DiastolicBP", "CholesterolTotal", 
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "UPDRS", 
    "MoCA", "FunctionalAssessment", "Tremor", "Rigidity", "Bradykinesia", 
    "PosturalInstability", "SpeechProblems", "SleepDisorders", "Constipation"
]

# Check if the model exists
model_path = "models/random_forest.pkl"
if not os.path.exists(model_path):
    print("\033[91m❌ ERROR: Model file not found! Train the model first using train.py.\033[0m")
    exit()

# Load the trained model
rf_model = joblib.load(model_path)

# Function to take user input and predict Parkinson's
def predict_parkinsons():
    print("\n🔍 Parkinson's Disease Prediction 🔍")
    print("-" * 50)
    input_data = []
    
    # Collect user input
    for feature in feature_names:
        while True:
            try:
                value = input(f"Enter value for {feature}: ")
                
                # Convert categorical features to int, numeric features to float
                if feature in ["Gender", "Ethnicity", "EducationLevel", "Smoking", "AlcoholConsumption", 
                               "PhysicalActivity", "DietQuality", "SleepQuality", "FamilyHistoryParkinsons", 
                               "TraumaticBrainInjury", "Hypertension", "Diabetes", "Depression", "Stroke", 
                               "Tremor", "Rigidity", "Bradykinesia", "PosturalInstability", "SpeechProblems", 
                               "SleepDisorders", "Constipation"]:
                    value = int(value)  # Categorical inputs as integers (0 or 1)
                else:
                    value = float(value)  # Numeric inputs as float
                
                input_data.append(value)
                break  # Exit loop if input is valid
            except ValueError:
                print("❌ Invalid input! Please enter a valid number.")
    
    # Convert input into DataFrame for model prediction
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    print("\n📌 Entered Patient Data:")
    print(input_df.to_string(index=False))
    print("-" * 50)
    
    # Predict using the trained model
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0]  # Get confidence scores
    
    # Display result
    print("\n🎯 Final Prediction:")
    print("\n📊 Confidence Scores:")
    print(f"  🔹 No Parkinson's: {probability[0] * 100:.2f}%")
    print(f"  🔸 Parkinson's: {probability[1] * 100:.2f}%")
    
    if prediction == 1:
        print(f"🛑 \033[91mThe model predicts that the patient HAS Parkinson's disease.\033[0m")
    else:
        print(f"✅ \033[92mThe model predicts that the patient DOES NOT HAVE Parkinson's disease.\033[0m")

    print("-" * 50)

# Run the prediction function
if __name__ == "__main__":
    predict_parkinsons()
