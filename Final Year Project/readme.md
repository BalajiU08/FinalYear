# 🧠 Parkinson's Disease Prediction System  

## 📌 Project Overview  
This project is a **machine learning-based system** that predicts whether a person has **Parkinson’s disease** based on various health and lifestyle factors. It uses a **Random Forest classifier**, trained on medical and demographic features, to make predictions and proves that it is better than the **KNN** for predicting the disease. 

---

## 🏗️ Project Structure  

📂 **Final Year Project**  
 ├── 📂 **data/** (Dataset storage)  
 │   ├── raw/  *(Original dataset)*  
 │   ├── processed/  *(Cleaned & preprocessed dataset)*  
 ├── 📂 **models/** (Trained models storage)  
 │   ├── random_forest.pkl *(Saved model)*  
 │   ├── knn.pkl *(Saved model for comparison)*  
 ├── 📂 **src/** (Code files)  
 │   ├── train.py *(Trains the ML models & saves them)*  
 │   ├── predict.py *(Loads the model & makes predictions)*  
 ├── 📜 requirements.txt *(List of dependencies to install)*  
 ├── 📜 README.md *(Project documentation & usage guide)*  

---

## ⚙️ Setup & Installation  

### 1️⃣ **Clone the Repository**  
```bash
git clone 
cd parkinsons-prediction

### 2️⃣ **Install Dependencies**
pip install -r requirements.txt

### 3️⃣ **Prepare the Dataset**
Place the preprocessed dataset inside data/processed/ as parkinsons_preprocessed.csv.

### 4️⃣ ***Train the Model***
Run the training script to train Random Forest and KNN models.
python src/train.py

Ensure the **models** folder exists
This will save the trained models inside the models/ directory.

### 5️⃣ Make Predictions
Run the prediction script to input patient details and get a prediction.
python src/predict.py

🏆 Features
✅ Random Forest Model: Achieves high accuracy (~90%)
✅ User Input System: Takes patient data from console input
✅ Model Comparison: KNN is used for benchmarking against Random Forest
✅ Data Visualization: Uses confusion matrix and classification reports
✅ Confidence Score: Displays model confidence for predictions

