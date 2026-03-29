from flask import Flask, jsonify, request
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from app import predict_single_stock, STOCK_CATEGORIES, DEFAULT_MODEL_DIR

flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        company = data.get('company')
        symbol = STOCK_CATEGORIES.get(company)
        if not symbol:
            return jsonify({'error': 'Company not found'}), 404
        result, err = predict_single_stock(symbol, DEFAULT_MODEL_DIR)
        if err:
            return jsonify({'error': err}), 500
        return jsonify({
            'company': company,
            'advice': result['advice'],
            'bullish_percentage': result['bullish_conf'],
            'bearish_percentage': result['bearish_conf'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@flask_app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))