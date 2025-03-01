from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Feature names (same order as in the form)
            feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                             "restecg", "thalach", "exang", "oldpeak", "slope", 
                             "ca", "thal"]

            # Get form data
            features = [float(request.form[field]) for field in feature_names]

            # Save raw user inputs for display
            user_data = features.copy()

            # Scale only relevant numerical features
            features[:5] = scaler.transform([features[:5]])[0]

            # Make prediction
            prediction = model.predict([features])[0]
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

            # Render result page with prediction & user data
            return render_template("result.html", result=result, user_data=user_data)

        except Exception as e:
            return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
