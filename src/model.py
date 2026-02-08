import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_URL_LENGTH, CHAR_VOCAB_SIZE, EMBEDDING_DIM,
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, LEARNING_RATE
)


def build_multimodal_cnn():
    """
    Build the Multi-Modal HF-CNN model for malicious website detection.

    Architecture:
    - Branch 1: Character-level 1D CNN for URL text
    - Branch 2: 2D CNN for numerical features reshaped as 4x4 images
    - Fusion: Concatenation + ANN classifier
    """

    # ================================================================
    # BRANCH 1: Text Feature Branch (CNN-1)
    # ================================================================
    text_input = Input(shape=(MAX_URL_LENGTH,), dtype='int32', name='text_input')

    # Character Embedding
    x_text = layers.Embedding(
        input_dim=CHAR_VOCAB_SIZE + 1,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_URL_LENGTH,
        name='char_embedding'
    )(text_input)

    # Conv1D Block 1
    x_text = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                           name='text_conv1d_1')(x_text)
    x_text = layers.BatchNormalization(name='text_bn_1')(x_text)
    x_text = layers.MaxPooling1D(pool_size=2, name='text_maxpool_1')(x_text)

    # Conv1D Block 2
    x_text = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                           name='text_conv1d_2')(x_text)
    x_text = layers.BatchNormalization(name='text_bn_2')(x_text)
    x_text = layers.MaxPooling1D(pool_size=2, name='text_maxpool_2')(x_text)

    # Conv1D Block 3
    x_text = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu',
                           name='text_conv1d_3')(x_text)
    x_text = layers.BatchNormalization(name='text_bn_3')(x_text)
    x_text = layers.GlobalMaxPooling1D(name='text_global_maxpool')(x_text)
    # Output: (batch, 32)

    # ================================================================
    # BRANCH 2: Numerical Feature Branch (CNN-2)
    # ================================================================
    image_input = Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        dtype='float32',
        name='image_input'
    )

    # Conv2D Block 1
    x_img = layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu',
                          name='img_conv2d_1')(image_input)
    x_img = layers.BatchNormalization(name='img_bn_1')(x_img)
    x_img = layers.MaxPooling2D(pool_size=(2, 2), padding='same',
                                name='img_maxpool_1')(x_img)

    # Conv2D Block 2
    x_img = layers.Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu',
                          name='img_conv2d_2')(x_img)
    x_img = layers.BatchNormalization(name='img_bn_2')(x_img)
    x_img = layers.GlobalAveragePooling2D(name='img_global_avgpool')(x_img)
    # Output: (batch, 64)

    # ================================================================
    # FUSION + ANN CLASSIFIER
    # ================================================================
    fused = layers.Concatenate(name='fusion_concat')([x_text, x_img])
    # Output: (batch, 96)

    # Dense Block 1
    fused = layers.Dense(256, activation='relu', name='fc_1')(fused)
    fused = layers.BatchNormalization(name='fc_bn_1')(fused)
    fused = layers.Dropout(0.5, name='fc_dropout_1')(fused)

    # Dense Block 2
    fused = layers.Dense(128, activation='relu', name='fc_2')(fused)
    fused = layers.BatchNormalization(name='fc_bn_2')(fused)
    fused = layers.Dropout(0.3, name='fc_dropout_2')(fused)

    # Dense Block 3
    fused = layers.Dense(64, activation='relu', name='fc_3')(fused)
    fused = layers.Dropout(0.2, name='fc_dropout_3')(fused)

    # Output Layer
    output = layers.Dense(1, activation='sigmoid', name='output')(fused)

    # Build and compile
    model = Model(
        inputs=[text_input, image_input],
        outputs=output,
        name='HF_CNN_Multimodal'
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR')
        ]
    )

    return model


if __name__ == '__main__':
    model = build_multimodal_cnn()
    model.summary()
