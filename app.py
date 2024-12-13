from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('cnn_model.h5')

# Serve the index.html file
@app.route('/')
def home():
    return send_from_directory(directory=os.getcwd(), path='index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the request
        data = request.get_json()
        input_features = np.array([data["features"]]).reshape(1, -1, 1)
        
        # Make prediction
        prediction = model.predict(input_features)
        predicted_class = int(prediction[0][0] > 0.5)  # Binary classification
        
        return jsonify({"prediction": predicted_class, "confidence": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Use the PORT environment variable or default to 5000
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
