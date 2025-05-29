import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import lime.lime_tabular

app = Flask(__name__)

datasets = {
    "Diabetes": "datasets/diabetes.csv",
    "Cancer": "datasets/cancer.csv",
    "Insurance": "datasets/insurance.csv"
}

def load_dataset(name):
    return pd.read_csv(datasets[name])

@app.route("/")
def index():
    return render_template("index.html", datasets=datasets.keys())

@app.route("/dataset", methods=["POST"])
def dataset():
    dataset_name = request.form["dataset"]
    df = load_dataset(dataset_name)
    columns = df.columns.tolist()
    preview = df.head().to_html(classes="table table-striped")
    return render_template("dataset.html", dataset_name=dataset_name, columns=columns, preview=preview, datasets=datasets.keys())

@app.route("/statistics", methods=["POST"])
def statistics():
    dataset_name = request.form["dataset"]
    df = load_dataset(dataset_name)
    stats = df.describe().to_html(classes="table table-striped")
    return render_template("statistics.html", dataset_name=dataset_name, stats=stats, datasets=datasets.keys())

@app.route("/graph", methods=["POST"])
def graph():
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    dataset_name = request.form["dataset"]
    column = request.form["column"]
    df = load_dataset(dataset_name)
    if column not in df.columns:
        return "Invalid column", 400

    plt.figure(figsize=(6, 4))
    df[column].hist(bins=20)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")

    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template("graph.html", dataset_name=dataset_name, column=column, graph_url=graph_url, datasets=datasets.keys())

@app.route("/explain", methods=["POST"])
def explain():
    dataset_name = request.form["dataset"]
    model = tf.keras.models.load_model(f"models/{dataset_name.lower()}_nn.h5", compile=False)
    df = load_dataset(dataset_name)

    target_col = "diabetes" if dataset_name == "Diabetes" else "Diagnosis" if dataset_name == "Cancer" else "charges"
    features_path = f"models/{dataset_name.lower()}_features.pkl"
    scaler_path = f"models/{dataset_name.lower()}_scaler.pkl"

    model_features = joblib.load(features_path)
    scaler = joblib.load(scaler_path)

    column_descriptions = {
        "Diabetes": {
            "gender": "Gender of the person (Female/Male/Other)",
            "age": "Age of the person",
            "hypertension": "History of hypertension (1 = yes, 0 = no)",
            "heart_disease": "History of heart disease (1 = yes, 0 = no)",
            "smoking_history": "Smoking history category",
            "bmi": "Body Mass Index",
            "HbA1c_level": "Hemoglobin A1c level",
            "blood_glucose_level": "Blood glucose level",
            "diabetes": "Whether the person has diabetes"
        },
        "Cancer": {
            "Age": "Age of the person",
            "Gender": "0 = Female, 1 = Male",
            "BMI": "Body Mass Index",
            "Smoking": "1 = smokes, 0 = does not",
            "GeneticRisk": "Genetic predisposition (1 = yes, 0 = no)",
            "PhysicalActivity": "Hours of physical activity per week",
            "AlcoholIntake": "Units of alcohol consumed per week",
            "CancerHistory": "Past cancer diagnosis (1 = yes, 0 = no)"
        },
        "Insurance": {
            "age": "Age of the insured person",
            "sex": "0 = Female, 1 = Male",
            "bmi": "Body Mass Index",
            "children": "Number of children covered by insurance",
            "smoker": "0 = no, 1 = yes",
            "region": "Region name",
            "charges": "Annual medical charges"
        }
    }

    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X)
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]
    X_scaled = scaler.transform(X)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=model_features,
        mode="classification" if dataset_name != "Insurance" else "regression"
    )

    sample_idx = 0
    lime_exp = lime_explainer.explain_instance(
        X_scaled[sample_idx],
        lambda x: model.predict(x).reshape(-1, 1) if dataset_name == "Insurance"
        else np.hstack([1 - model.predict(x), model.predict(x)])
    )

    lime_html = lime_exp.as_html()

    return render_template("explain.html",
                           dataset_name=dataset_name,
                           lime_html=lime_html,
                           column_info=column_descriptions.get(dataset_name, {}),
                           datasets=datasets.keys())

@app.route("/try_model", methods=["GET", "POST"])
def try_model():
    if request.method == "GET":
        dataset_name = request.args.get("dataset", "Diabetes")
        features = joblib.load(f"models/{dataset_name.lower()}_features.pkl")

        acc_path = f"models/{dataset_name.lower()}_accuracy.txt"
        accuracy = None
        if os.path.exists(acc_path):
            with open(acc_path, "r") as f:
                accuracy = f.read()

        return render_template("try_model.html",
                               dataset_name=dataset_name,
                               features=features,
                               accuracy=accuracy,
                               datasets=datasets.keys())

    dataset_name = request.form["dataset"]
    model = tf.keras.models.load_model(f"models/{dataset_name.lower()}_nn.h5")
    scaler = joblib.load(f"models/{dataset_name.lower()}_scaler.pkl")
    features = joblib.load(f"models/{dataset_name.lower()}_features.pkl")

    form_data = dict(request.form)

    # 🔁 Encoding handling
    if dataset_name == "Diabetes":
        gender_str = form_data.pop("gender", "")
        smoking_str = form_data.pop("smoking_history", "")
        gender_enc = joblib.load("models/diabetes_gender_encoder.pkl")
        smoking_enc = joblib.load("models/diabetes_smoking_encoder.pkl")
        form_data["gender"] = gender_enc.transform([gender_str])[0]
        form_data["smoking_history"] = smoking_enc.transform([smoking_str])[0]

    if dataset_name == "Insurance":
        sex_str = form_data.pop("sex", "")
        smoker_str = form_data.pop("smoker", "")
        region_str = form_data.pop("region", "")
        sex_enc = joblib.load("models/insurance_sex_encoder.pkl")
        smoker_enc = joblib.load("models/insurance_smoker_encoder.pkl")
        region_enc = joblib.load("models/insurance_region_encoder.pkl")
        form_data["sex"] = sex_enc.transform([sex_str])[0]
        form_data["smoker"] = smoker_enc.transform([smoker_str])[0]
        form_data["region"] = region_enc.transform([region_str])[0]

    input_values = []
    for f in features:
        val = form_data.get(f, 0)
        input_values.append(float(val))

    X_input = np.array(input_values).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    if dataset_name == "Insurance":
        prediction = model.predict(X_scaled)[0][0]
        prediction_output = f"${prediction:.2f}"
        classification = None
        confidence = None
    else:
        probability = model.predict(X_scaled)[0][0]
        prediction_output = "Has" if probability >= 0.5 else "Does not have"
        confidence = round(probability * 100, 2) if prediction_output == "Has" else round((1 - probability) * 100, 2)
        classification = "Diabetes" if dataset_name == "Diabetes" else "Cancer"

    df = pd.read_csv(f"datasets/{dataset_name.lower()}.csv")
    target_col = "diabetes" if dataset_name == "Diabetes" else "Diagnosis" if dataset_name == "Cancer" else "charges"
    X_df = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

    for col in features:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df[features]
    X_df_scaled = scaler.transform(X_df.values)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_df_scaled,
        feature_names=features,
        mode="classification" if dataset_name != "Insurance" else "regression"
    )

    lime_exp = explainer.explain_instance(
        X_scaled[0],
        lambda x: model.predict(x).reshape(-1, 1) if dataset_name == "Insurance"
        else np.hstack([1 - model.predict(x), model.predict(x)])
    )

    lime_html = lime_exp.as_html()

    top_features = lime_exp.as_list()
    explanation_text = ", ".join([desc for desc, _ in top_features[:4]])

    if dataset_name == "Insurance":
        final_reasoning = f"Receives {prediction_output} because {explanation_text}."
    else:
        final_reasoning = f"{prediction_output} {classification.lower()} because {explanation_text}."

    return render_template("try_model.html",
                           dataset_name=dataset_name,
                           features=features,
                           confidence=confidence,
                           classification=classification,
                           prediction_output=prediction_output,
                           explanation_text=final_reasoning,
                           lime_html=lime_html,
                           datasets=datasets.keys())


if __name__ == "__main__":
    app.run(debug=True)
