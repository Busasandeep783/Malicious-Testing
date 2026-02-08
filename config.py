import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'url_dataset.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# ---- Model Artifact Paths ----
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'char_tokenizer.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.pkl')

# ---- Text Branch ----
MAX_URL_LENGTH = 200
CHAR_VOCAB_SIZE = 128
EMBEDDING_DIM = 64

# ---- Image Branch ----
IMAGE_HEIGHT = 4
IMAGE_WIDTH = 4
IMAGE_CHANNELS = 1

# ---- Feature Columns ----
# 16 numerical features from the phishing URL dataset
NUMERICAL_FEATURE_COLS = [
    'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth',
    'Redirection', 'https_Domain', 'TinyURL', 'Prefix/Suffix',
    'DNS_Record', 'Web_Traffic', 'Domain_Age', 'Domain_End',
    'iFrame', 'Mouse_Over', 'Right_Click', 'Web_Forwards'
]

# Column names
URL_COL = 'Domain'
TARGET_COL = 'Label'

# ---- Training ----
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42
