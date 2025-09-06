print(">>> app.py started")
from flask import Flask, request, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load your trained model
with open("bodyfat_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features from form
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])  # reshape for model

        # Make prediction
        prediction = model.predict(features_array)[0]

        return render_template("index.html", prediction_text=f"Predicted Body Fat: {prediction:.2f}%")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
    