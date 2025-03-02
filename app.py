from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder="templates")

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        features = np.array(float_features).reshape(1, -1)

        features = scaler.transform(features)

        confidence = model.predict_proba(features)[0][1] * 100  

        if confidence > 70:
            message = "⚠️ High Risk: You should consult a cardiologist immediately."
            advice = "Maintain a healthy diet, exercise regularly, and monitor your blood pressure."
        elif confidence > 40:
            message = "⚠️ Medium Risk: You are at risk of cardiovascular disease."
            advice = "Eat more fruits and vegetables, reduce salt intake, and stay active daily."
        else:
            message = "✅ Low Risk: Congratulations! You are healthy."
            advice = "Keep up your healthy habits and maintain a balanced lifestyle."

        result_text = f"You are {confidence:.2f}% likely to be suffering from cardiovascular disease."
        return render_template("index.html", data=result_text, message=message, advice=advice)

    except Exception as e:
        return render_template("index.html", data=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
