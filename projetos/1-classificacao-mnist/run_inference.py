import os

import numpy as np
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(SCRIPT_DIR, "model.tflite")
N_SAMPLES = 10


def main():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]

    correct = 0
    print(f"Inferência de {N_SAMPLES} amostras com model.tflite")
    for index in range(N_SAMPLES):
        sample = x_test[index : index + 1].astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details["index"])[0]
        predicted = int(np.argmax(scores))
        expected = int(y_test[index])
        correct += predicted == expected
        print(
            f"Amostra {index + 1:02d}: predito={predicted} | "
            f"real={expected} | confiança={float(np.max(scores)):.4f}"
        )

    print(f"Acertos na amostra: {correct}/{N_SAMPLES}")


if __name__ == "__main__":
    main()
