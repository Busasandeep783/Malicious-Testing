from flask import Flask, render_template, request, jsonify
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import MaliciousURLPredictor

app = Flask(__name__)

# Load model once at startup
print("Loading model...")
predictor = MaliciousURLPredictor()
examples = predictor.get_example_urls(n=5)
print("Model loaded successfully!")


@app.route('/')
def index():
    """Render the home page with URL input form and examples."""
    return render_template('index.html', examples=examples)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept a URL, run prediction, return result page."""
    url = request.form.get('url', '').strip()

    if not url:
        return render_template('index.html', examples=examples,
                               error='Please enter a URL.')

    try:
        result = predictor.predict(url)
        return render_template('result.html', url=url, result=result)
    except Exception as e:
        return render_template('index.html', examples=examples,
                               error=f'Error analyzing URL: {str(e)}')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic access."""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" field'}), 400

    try:
        result = predictor.predict(data['url'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
