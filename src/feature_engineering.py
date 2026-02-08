import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_URL_LENGTH, CHAR_VOCAB_SIZE, NUMERICAL_FEATURE_COLS,
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
)


def prepare_text_features(urls, tokenizer=None, fit=True):
    """
    Convert URL/domain strings to character-level integer sequences.

    Each character is mapped to an integer index.
    Sequences are padded/truncated to MAX_URL_LENGTH.

    Returns:
        padded_sequences: np.array of shape (n_samples, MAX_URL_LENGTH)
        tokenizer: Fitted Tokenizer object
    """
    # Convert to list of strings
    urls = [str(u) for u in urls]

    if fit:
        tokenizer = Tokenizer(char_level=True, num_words=CHAR_VOCAB_SIZE,
                              oov_token='<OOV>')
        tokenizer.fit_on_texts(urls)

    sequences = tokenizer.texts_to_sequences(urls)
    padded = pad_sequences(sequences, maxlen=MAX_URL_LENGTH,
                           padding='post', truncating='post')
    return padded, tokenizer


def prepare_numerical_features(df, scaler=None, fit=True):
    """
    Scale numerical features and reshape into a 2D image grid.

    Steps:
    1. Extract the 16 numerical feature columns
    2. StandardScaler normalization
    3. Reshape each sample to (4, 4, 1) -- a single-channel 4x4 "image"

    Returns:
        images: np.array of shape (n_samples, 4, 4, 1)
        scaler: Fitted StandardScaler
    """
    X_num = df[NUMERICAL_FEATURE_COLS].values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = scaler.transform(X_num)

    n_samples = X_num.shape[0]
    target_size = IMAGE_HEIGHT * IMAGE_WIDTH  # 16
    n_features = X_num.shape[1]  # 16

    # Pad if needed (16 features fit exactly into 4x4)
    if n_features < target_size:
        padding = np.zeros((n_samples, target_size - n_features))
        X_num = np.hstack([X_num, padding])
    elif n_features > target_size:
        X_num = X_num[:, :target_size]

    # Reshape to (n_samples, 4, 4, 1)
    X_images = X_num.reshape(n_samples, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    return X_images, scaler
