from flask import Flask, request, jsonify
import joblib
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)
model = joblib.load('phising_detector_model.py')  # Load the trained model

def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'has_https': 1 if parsed_url.scheme == 'https' else 0,
    }
    return pd.Series(features)

@app.route('/api/detect', methods=['POST'])
def detect_phishing():
    data = request.json
    url = data['url']
    features = extract_features(url)
    prediction = model.predict([features])[0]
    return jsonify({'is_phishing': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
