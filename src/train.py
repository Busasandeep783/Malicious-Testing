import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import prepare_text_features, prepare_numerical_features
from src.model import build_multimodal_cnn


def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- Step 1: Load and Clean Data ----
    print("=" * 60)
    print("[1/6] Loading and cleaning data...")
    print("=" * 60)
    df = load_data()
    df = clean_data(df)

    # ---- Step 2: Prepare Features ----
    print("\n" + "=" * 60)
    print("[2/6] Preparing text and numerical features...")
    print("=" * 60)

    urls = df[URL_COL].values
    y = df[TARGET_COL].values.astype(np.float32)

    # Text features (character-level URL encoding)
    X_text, tokenizer = prepare_text_features(urls, fit=True)
    print(f"  Text features shape: {X_text.shape}")

    # Numerical features (reshaped as 4x4 images)
    X_numerical, scaler = prepare_numerical_features(df, fit=True)
    print(f"  Numerical features shape: {X_numerical.shape}")

    # ---- Step 3: Train/Val/Test Split ----
    print("\n" + "=" * 60)
    print("[3/6] Splitting data (70/15/15)...")
    print("=" * 60)

    indices = np.arange(len(y))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    X_train_text = X_text[idx_train]
    X_val_text = X_text[idx_val]
    X_test_text = X_text[idx_test]

    X_train_num = X_numerical[idx_train]
    X_val_num = X_numerical[idx_val]
    X_test_num = X_numerical[idx_test]

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Train positive ratio: {y_train.mean():.3f}")

    # ---- Compute Class Weights (mild balancing) ----
    pos_ratio = y_train.mean()
    neg_ratio = 1 - pos_ratio
    # Use sqrt-balanced weights for mild correction
    class_weight_dict = {
        0: np.sqrt(1.0 / (2 * neg_ratio)) if neg_ratio > 0 else 1.0,
        1: np.sqrt(1.0 / (2 * pos_ratio)) if pos_ratio > 0 else 1.0
    }
    print(f"  Class weights: {class_weight_dict}")

    # ---- Step 4: Build Model ----
    print("\n" + "=" * 60)
    print("[4/6] Building multi-modal CNN model...")
    print("=" * 60)

    model = build_multimodal_cnn()
    model.summary()

    # ---- Step 5: Setup Callbacks ----
    callbacks = [
        EarlyStopping(
            monitor='val_auc_roc',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_auc_roc',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(os.path.join(MODEL_DIR, 'training_log.csv'), append=False),
    ]

    # ---- Step 6: Train Model ----
    print("\n" + "=" * 60)
    print("[5/6] Training model...")
    print("=" * 60)

    history = model.fit(
        x=[X_train_text, X_train_num],
        y=y_train,
        validation_data=([X_val_text, X_val_num], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # ---- Save Artifacts ----
    print("\n" + "=" * 60)
    print("[6/6] Saving model artifacts...")
    print("=" * 60)

    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  Saved tokenizer: {TOKENIZER_PATH}")

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler: {SCALER_PATH}")

    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"  Saved training history: {HISTORY_PATH}")

    # Save test data for evaluation
    np.save(os.path.join(PROCESSED_DIR, 'X_test_text.npy'), X_test_text)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_numerical.npy'), X_test_num)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    print(f"  Saved test data to: {PROCESSED_DIR}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return model, history


if __name__ == '__main__':
    main()
