import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *

# =========================
# SETTINGS
# =========================

IMG_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 20

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

# =========================
# DATASET LOADER
# =========================

def load_data():

    images = []
    masks = []

    image_files = sorted(os.listdir(IMAGE_DIR))

    for file in image_files:

        img_path = os.path.join(IMAGE_DIR, file)

        mask_path = os.path.join(
            MASK_DIR,
            file
        )

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE)
        )

        mask = mask.astype(np.float32) / 255.0

        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    return images, masks

# =========================
# METRICS
# =========================

def dice_coef(y_true, y_pred):

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + 1
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + 1
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(
        y_true,
        y_pred
    )

# =========================
# U-NET
# =========================

def conv_block(x, filters):

    x = Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    return x

def build_unet():

    inputs = Input(
        (IMG_SIZE, IMG_SIZE, 3)
    )

    c1 = conv_block(inputs, 16)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = MaxPooling2D()(c3)

    c4 = conv_block(p3, 128)
    p4 = MaxPooling2D()(c4)

    bn = conv_block(p4, 128)

    u1 = UpSampling2D()(bn)
    u1 = concatenate([u1, c4])
    u1 = conv_block(u1, 256)

    u2 = UpSampling2D()(u1)
    u2 = concatenate([u2, c3])
    u2 = conv_block(u2, 128)

    u3 = UpSampling2D()(u2)
    u3 = concatenate([u3, c2])
    u3 = conv_block(u3, 64)

    u4 = UpSampling2D()(u3)
    u4 = concatenate([u4, c1])
    u4 = conv_block(u4, 32)

    outputs = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(u4)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss=dice_loss,
        metrics=[
            dice_coef,
            "accuracy"
        ]
    )

    return model

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    os.makedirs(
        "models",
        exist_ok=True
    )

    print("Loading dataset...")

    X, Y = load_data()

    print("Images:", X.shape)
    print("Masks :", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    model = build_unet()

    model.summary()

    checkpoint = ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    earlystop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        factor=0.5,
        verbose=1
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(
            X_test,
            Y_test
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpoint,
            earlystop,
            reduce_lr
        ]
    )

    print("\nTraining Complete")

    results = model.evaluate(
        X_test,
        Y_test
    )

    print(
        "\nFinal Dice Score:",
        results[1]
    )