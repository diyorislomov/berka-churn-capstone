from flask import Flask, request, jsonify
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_churn, load_model

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON body provided'}), 400
        result = predict_churn(data)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal error', 'details': str(e)}), 500

if __name__ == '__main__':
    load_model()  # warm up on startup so first request isn't slow
    app.run(debug=True, port=5000)