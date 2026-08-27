from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(_name_)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "AI Financial Performance Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'performance_score': float(prediction[0])})
if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)
