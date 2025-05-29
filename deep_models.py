import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("models", exist_ok=True)

datasets = {
    "Diabetes": {
        "path": "datasets/diabetes.csv",
        "target": "diabetes",
        "type": "classification"
    },
    "Cancer": {
        "path": "datasets/cancer.csv",
        "target": "Diagnosis",
        "type": "classification"
    },
    "Insurance": {
        "path": "datasets/insurance.csv",
        "target": "charges",
        "type": "regression"
    }
}

for name, config in datasets.items():
    print(f"🚀 Training model for {name}")
    df = pd.read_csv(config["path"])
    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]

    if name == "Diabetes":
        gender_enc = LabelEncoder()
        smoking_enc = LabelEncoder()

        X["gender"] = gender_enc.fit_transform(X["gender"])
        X["smoking_history"] = smoking_enc.fit_transform(X["smoking_history"])

        joblib.dump(gender_enc, "models/diabetes_gender_encoder.pkl")
        joblib.dump(smoking_enc, "models/diabetes_smoking_encoder.pkl")

    if name == "Insurance":
        sex_enc = LabelEncoder()
        smoker_enc = LabelEncoder()
        region_enc = LabelEncoder()

        X["sex"] = sex_enc.fit_transform(X["sex"])
        X["smoker"] = smoker_enc.fit_transform(X["smoker"])
        X["region"] = region_enc.fit_transform(X["region"])

        joblib.dump(sex_enc, "models/insurance_sex_encoder.pkl")
        joblib.dump(smoker_enc, "models/insurance_smoker_encoder.pkl")
        joblib.dump(region_enc, "models/insurance_region_encoder.pkl")

    X = pd.get_dummies(X)
    features = X.columns.tolist()

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid' if config["type"] == "classification" else 'linear')
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy() if config["type"] == "classification" else tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        metrics=['accuracy'] if config["type"] == "classification" else ['mae']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train_scaled, y_train,
              validation_data=(X_val_scaled, y_val),
              epochs=200,
              batch_size=32,
              callbacks=[early_stop],
              verbose=1)

    results = model.evaluate(X_test_scaled, y_test)
    performance = round(results[1] * 100, 2) if config["type"] == "classification" else round(results[1], 2)

    model.save(f"models/{name.lower()}_nn.h5")
    joblib.dump(scaler, f"models/{name.lower()}_scaler.pkl")
    joblib.dump(features, f"models/{name.lower()}_features.pkl")

    with open(f"models/{name.lower()}_accuracy.txt", "w") as f:
        f.write(str(performance))

    print(f"✅ {name} model performance: {performance}")

print("🎉 All models trained and saved.")
