import os

import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(SCRIPT_DIR, "model.h5")
TFLITE_PATH = os.path.join(SCRIPT_DIR, "model.tflite")


def main():
    model = tf.keras.models.load_model(H5_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as output_file:
        output_file.write(tflite_model)

    h5_size = os.path.getsize(H5_PATH)
    tflite_size = os.path.getsize(TFLITE_PATH)
    reduction = (1 - tflite_size / h5_size) * 100

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    print(f"Tamanho model.h5: {h5_size / 1024:.1f} KB")
    print(f"Tamanho model.tflite: {tflite_size / 1024:.1f} KB")
    print(f"Redução de tamanho: {reduction:.1f}%")
    print("model.tflite carregado e tensores alocados com sucesso.")


if __name__ == "__main__":
    main()
