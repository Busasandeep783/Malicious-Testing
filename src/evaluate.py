import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score
)
import tensorflow as tf
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_SAVE_PATH, PROCESSED_DIR, HISTORY_PATH, PLOTS_DIR


def evaluate_model(model, X_test_text, X_test_num, y_test, output_dir=PLOTS_DIR):
    """Comprehensive evaluation with metrics and plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    y_pred_prob = model.predict([X_test_text, X_test_num]).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # ---- Metrics ----
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec_vals, prec_vals)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall_val:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")
    print(f"AUC-PR:    {pr_auc:.4f}")

    # ---- Plot 1: Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"\nSaved: confusion_matrix.png")

    # ---- Plot 2: ROC Curve ----
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    print(f"Saved: roc_curve.png")

    # ---- Plot 3: Precision-Recall Curve ----
    plt.figure(figsize=(8, 6))
    plt.plot(rec_vals, prec_vals, color='green', lw=2,
             label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()
    print(f"Saved: precision_recall_curve.png")

    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall_val,
        'f1': f1, 'mcc': mcc, 'auc_roc': roc_auc, 'auc_pr': pr_auc
    }


def plot_training_history(history_path=HISTORY_PATH, output_dir=PLOTS_DIR):
    """Plot training and validation curves."""
    os.makedirs(output_dir, exist_ok=True)

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Model Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='red')
    axes[0, 1].set_title('Model Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC-ROC
    axes[1, 0].plot(history['auc_roc'], label='Train AUC-ROC', color='blue')
    axes[1, 0].plot(history['val_auc_roc'], label='Val AUC-ROC', color='red')
    axes[1, 0].set_title('AUC-ROC', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision and Recall
    axes[1, 1].plot(history['precision'], label='Train Precision', color='blue')
    axes[1, 1].plot(history['val_precision'], label='Val Precision', color='red')
    axes[1, 1].plot(history['recall'], label='Train Recall', color='green')
    axes[1, 1].plot(history['val_recall'], label='Val Recall', color='orange')
    axes[1, 1].set_title('Precision & Recall', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    print(f"Saved: training_history.png")


if __name__ == '__main__':
    # Load model and test data
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    X_test_text = np.load(os.path.join(PROCESSED_DIR, 'X_test_text.npy'))
    X_test_num = np.load(os.path.join(PROCESSED_DIR, 'X_test_numerical.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

    results = evaluate_model(model, X_test_text, X_test_num, y_test)
    plot_training_history()
