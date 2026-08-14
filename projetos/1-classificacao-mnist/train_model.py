import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.h5")
VALIDATION_SIZE = 6_000
MAX_EPOCHS = 12
BATCH_SIZE = 128


def load_data():
    (x_train_all, y_train_all), _ = tf.keras.datasets.mnist.load_data()
    x_train_all = x_train_all.astype("float32") / 255.0
    x_train_all = x_train_all[..., tf.newaxis]

    x_train = x_train_all[:-VALIDATION_SIZE]
    y_train = y_train_all[:-VALIDATION_SIZE]
    x_val = x_train_all[-VALIDATION_SIZE:]
    y_val = y_train_all[-VALIDATION_SIZE:]
    return x_train, y_train, x_val, y_val


def build_model():
    return models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_edge_cnn",
    )


def main():
    tf.keras.utils.set_random_seed(42)
    x_train, y_train, x_val, y_val = load_data()

    with tf.device("/CPU:0"):
        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=2,
        )
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)

    print(f"Perda final de validação: {val_loss:.4f}")
    print(f"Acurácia final de validação: {val_accuracy:.4%}")
    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
