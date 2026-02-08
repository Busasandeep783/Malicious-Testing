import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from urllib.parse import urlparse
import requests
import socket
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class MaliciousURLPredictor:
    """
    Predictor for malicious website detection.
    Uses the trained Multi-Modal CNN for dataset URLs (with known features)
    and a Random Forest fallback for arbitrary live URLs.
    """

    def __init__(self):
        # Load CNN model
        self.cnn_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

        with open(TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load Random Forest for live predictions
        rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
        with open(rf_path, 'rb') as f:
            self.rf_model = pickle.load(f)

        # Load dataset for known domains
        self.dataset = pd.read_csv(DATA_PATH)
        self.known_domains = set(self.dataset[URL_COL].str.lower().values)

    def get_example_urls(self, n=5):
        """Get example benign and malicious URLs from the dataset."""
        df = self.dataset.sample(frac=1, random_state=42)
        benign = df[df[TARGET_COL] == 0].head(n)
        malicious = df[df[TARGET_COL] == 1].head(n)
        return {
            'benign': benign[URL_COL].tolist(),
            'malicious': malicious[URL_COL].tolist()
        }

    def predict_from_dataset(self, domain):
        """Predict using CNN model with actual dataset features."""
        # Find the domain in dataset
        matches = self.dataset[self.dataset[URL_COL].str.lower() == domain.lower()]
        if matches.empty:
            return None

        row = matches.iloc[0]

        # Prepare text input
        seq = self.tokenizer.texts_to_sequences([row[URL_COL]])
        X_text = pad_sequences(seq, maxlen=MAX_URL_LENGTH,
                               padding='post', truncating='post')

        # Prepare numerical input
        num_values = [float(row[col]) for col in NUMERICAL_FEATURE_COLS]
        num_vector = np.array([num_values], dtype=np.float32)
        num_scaled = self.scaler.transform(num_vector)
        X_num = num_scaled.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

        # CNN prediction
        prob = float(self.cnn_model.predict([X_text, X_num], verbose=0)[0][0])
        prediction = int(prob >= 0.5)

        features = {col: int(row[col]) for col in NUMERICAL_FEATURE_COLS}

        return {
            'prediction': prediction,
            'label': 'Malicious' if prediction == 1 else 'Benign',
            'confidence': prob if prediction == 1 else 1 - prob,
            'probability_malicious': prob,
            'features': features,
            'model_used': 'Multi-Modal CNN (Dual-Branch)'
        }

    def extract_live_features(self, url):
        """Extract features from a live URL for RF prediction."""
        features = {}

        if not url.startswith(('http://', 'https://')):
            full_url = 'http://' + url
        else:
            full_url = url

        parsed = urlparse(full_url)
        domain = parsed.netloc or parsed.path.split('/')[0]

        features['Have_IP'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        features['Have_At'] = 1 if '@' in url else 0
        features['URL_Length'] = 1 if len(url) > 54 else 0
        features['URL_Depth'] = parsed.path.count('/') if parsed.path else 0
        features['Redirection'] = 1 if '//' in parsed.path else 0
        features['https_Domain'] = 0

        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
                       'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc']
        features['TinyURL'] = 1 if any(s in domain.lower() for s in shorteners) else 0
        features['Prefix/Suffix'] = 1 if '-' in domain else 0
        features['DNS_Record'] = 0

        try:
            socket.gethostbyname(domain)
            features['Web_Traffic'] = 1
        except Exception:
            features['Web_Traffic'] = 0

        features['Domain_Age'] = 0
        features['Domain_End'] = 1

        try:
            import whois
            from datetime import datetime, timezone
            w = whois.whois(domain)
            now = datetime.now(timezone.utc)

            if w.creation_date:
                creation = w.creation_date
                if isinstance(creation, list):
                    creation = creation[0]
                if creation.tzinfo is None:
                    creation = creation.replace(tzinfo=timezone.utc)
                features['Domain_Age'] = 1 if (now - creation).days > 365 else 0

            if w.expiration_date:
                expiry = w.expiration_date
                if isinstance(expiry, list):
                    expiry = expiry[0]
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                features['Domain_End'] = 1 if (expiry - now).days > 180 else 0
        except Exception:
            pass

        features['iFrame'] = 0
        features['Mouse_Over'] = 0
        features['Right_Click'] = 1
        features['Web_Forwards'] = 0

        try:
            resp = requests.head(full_url, timeout=5, allow_redirects=True)
            features['Web_Forwards'] = 1 if len(resp.history) > 2 else 0
        except Exception:
            pass

        return features

    def predict(self, url):
        """
        Predict if a URL is malicious.
        Uses CNN for known dataset domains, RF for unknown live URLs.
        """
        # Strip any protocol prefix for domain matching
        domain = url.strip()
        if domain.startswith(('http://', 'https://')):
            parsed = urlparse(domain)
            domain = parsed.netloc or parsed.path.split('/')[0]

        # Try CNN prediction for known domains
        if domain.lower() in self.known_domains:
            result = self.predict_from_dataset(domain)
            if result is not None:
                return result

        # Fall back to RF for unknown domains
        features = self.extract_live_features(url)
        num_vector = np.array(
            [[features[col] for col in NUMERICAL_FEATURE_COLS]],
            dtype=np.float32
        )

        rf_proba = self.rf_model.predict_proba(num_vector)[0]
        prob = float(rf_proba[1])
        prediction = int(prob >= 0.5)

        return {
            'prediction': prediction,
            'label': 'Malicious' if prediction == 1 else 'Benign',
            'confidence': prob if prediction == 1 else 1 - prob,
            'probability_malicious': prob,
            'features': features,
            'model_used': 'Random Forest (Live Analysis)'
        }


if __name__ == '__main__':
    predictor = MaliciousURLPredictor()

    print("=== Known Dataset URLs (CNN Model) ===")
    for url in ['extratorrent.cc', 'medium.com', 'slashdot.org',
                'sertyxese.myfreesites.net', 'u704893oyf.ha004.t.justns.ru']:
        result = predictor.predict(url)
        print(f"\n  {url}: {result['label']} "
              f"(confidence={result['confidence']:.4f}, "
              f"model={result['model_used']})")

    print("\n=== Live URLs (RF Model) ===")
    for url in ['google.com', 'bit.ly/suspicious', '192.168.1.1/login']:
        result = predictor.predict(url)
        print(f"\n  {url}: {result['label']} "
              f"(confidence={result['confidence']:.4f}, "
              f"model={result['model_used']})")
