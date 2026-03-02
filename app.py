from flask import Flask, render_template, request, jsonify
import numpy as np
import h5py
import os

app = Flask(__name__)

# A simple Numpy implementation of the neural network
class NumpyModel:
    def __init__(self, h5_path):
        self.weights = []
        self.biases = []
        
        with h5py.File(h5_path, 'r') as f:
            # Dense 1
            w1 = f['/model_weights/dense/sequential/dense/kernel'][:]
            b1 = f['/model_weights/dense/sequential/dense/bias'][:]
            self.weights.append(w1)
            self.biases.append(b1)
            
            # Dense 2
            w2 = f['/model_weights/dense_1/sequential/dense_1/kernel'][:]
            b2 = f['/model_weights/dense_1/sequential/dense_1/bias'][:]
            self.weights.append(w2)
            self.biases.append(b2)
            
            # Dense 3
            w3 = f['/model_weights/dense_2/sequential/dense_2/kernel'][:]
            b3 = f['/model_weights/dense_2/sequential/dense_2/bias'][:]
            self.weights.append(w3)
            self.biases.append(b3)
            
    def relu(self, x):
        return np.maximum(0, x)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def predict(self, X):
        # Layer 1
        h1 = self.relu(np.dot(X, self.weights[0]) + self.biases[0])
        # Layer 2
        h2 = self.relu(np.dot(h1, self.weights[1]) + self.biases[1])
        # Output Layer
        out = self.sigmoid(np.dot(h2, self.weights[2]) + self.biases[2])
        return out

# Load the AI model
model_path = 'diabetes_model.h5'
try:
    if os.path.exists(model_path):
        model = NumpyModel(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Error: Could not find {model_path}.")
        model = None
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    return jsonify({"status": "AI Model Server is running natively!"})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    if model is None:
        return jsonify({'error': 'AI Model is not loaded.'}), 500

    try:
        data = request.get_json()
        
        # Extract features from the form data string payload and convert to float
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('bloodPressure', 0)),
            float(data.get('skinThickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('dpf', 0)),
            float(data.get('age', 0))
        ]
        
        # Reshape the input array for the model
        input_data = np.array(features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data)
        probability = float(prediction[0][0]) 
        
        result = "Diabetic" if probability >= 0.5 else "Non-Diabetic"
        
        return jsonify({
            'result': result,
            'probability': probability
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the server
    app.run(debug=True, port=5000)
