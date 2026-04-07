import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Carregamento do dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalização dos dados (valores entre 0 e 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Expansão de dimensão para adicionar o canal de cor (necessário para Conv2D)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# 2. Construção do modelo CNN (Simples para Edge AI, máximo de 3 camadas conv)
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compilação do modelo
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# 3. Treinamento do modelo (Número limitado de épocas para execução rápida em CI)
print("Iniciando o treinamento do modelo...")
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1)

# 4. Exibição da acurácia final no terminal
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final no conjunto de teste: {test_acc:.4f}")

# 5. Salvamento do modelo treinado no formato Keras (.h5)
model.save("model.h5")
print("Modelo salvo como model.h5")
