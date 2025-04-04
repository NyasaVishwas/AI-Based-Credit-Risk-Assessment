from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the scaler and model (adjust path if needed)
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/random_forest.pkl")  # or "xgboost.pkl"

@app.route('/')
def home():
    return "Welcome to the Credit Risk Assessment API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Get feature list from incoming JSON
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
