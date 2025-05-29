import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

datasets = {
    "Diabetes": "datasets/diabetes.csv",
    "Cancer": "datasets/cancer.csv",
    "Insurance": "datasets/insurance.csv"
}

def train_and_save_model(dataset_name, model_path, features_path):
    df = pd.read_csv(datasets[dataset_name])

    if dataset_name == "Diabetes":
        X = df.drop(columns=["diabetes"])
        y = df["diabetes"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif dataset_name == "Cancer":
        X = df.drop(columns=["Diagnosis"])
        y = df["Diagnosis"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        X = df.drop(columns=["charges"])
        y = df["charges"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # Save the trained model and feature names
    joblib.dump(model, model_path)
    joblib.dump(list(X_train.columns), features_path)  # Save feature names

    print(f"✅ Model saved: {model_path}")
    print(f"✅ Feature names saved: {features_path}")

# Train and save models
train_and_save_model("Diabetes", "models/diabetes_model.pkl", "models/diabetes_features.pkl")
train_and_save_model("Cancer", "models/cancer_model.pkl", "models/cancer_features.pkl")
train_and_save_model("Insurance", "models/insurance_model.pkl", "models/insurance_features.pkl")
