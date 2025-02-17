from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/prediction", methods=['POST'])
def prediction():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)
        
        # Extract input data from JSON
        age = data['age']
        height = data['height']
        weight = data['weight']
        num_surgeries = data['num_surgeries']
        diabetes = data['diabetes']
        bp_problems = data['bp_problems']
        transplants = data['transplants']
        chronic_diseases = data['chronic_diseases']
        allergies = data['allergies']
        cancer_history = data['cancer_history']

        # Create input DataFrame
        input_data = pd.DataFrame({
            "Age": [age],
            "Diabetes": [diabetes],
            "BloodPressureProblems": [bp_problems],
            "AnyTransplants": [transplants],
            "AnyChronicDiseases": [chronic_diseases],
            "Height": [height],
            "Weight": [weight],
            "KnownAllergies": [allergies],
            "HistoryOfCancerInFamily": [cancer_history],
            "NumberOfMajorSurgeries": [num_surgeries]
        })

        # Apply standard scaling to numerical features
        num_features = ["Age", "Height", "Weight"]
        input_data[num_features] = scaler.transform(input_data[num_features])

        # Predict Premium Price
        prediction = model.predict(input_data)[0]

        # Return the prediction as JSON
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        # Get form data
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        num_surgeries = int(request.form["num_surgeries"])
        diabetes = int(request.form["diabetes"])
        bp_problems = int(request.form["bp_problems"])
        transplants = int(request.form["transplants"])
        chronic_diseases = int(request.form["chronic_diseases"])
        allergies = int(request.form["allergies"])
        cancer_history = int(request.form["cancer_history"])

        # Create input DataFrame
        input_data = pd.DataFrame({
            "Age": [age],
            "Diabetes": [diabetes],
            "BloodPressureProblems": [bp_problems],
            "AnyTransplants": [transplants],
            "AnyChronicDiseases": [chronic_diseases],
            "Height": [height],
            "Weight": [weight],
            "KnownAllergies": [allergies],
            "HistoryOfCancerInFamily": [cancer_history],
            "NumberOfMajorSurgeries": [num_surgeries]
        })

        # Apply standard scaling to numerical features
        num_features = ["Age", "Height", "Weight"]
        input_data[num_features] = scaler.transform(input_data[num_features])

        # Predict Premium Price
        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)