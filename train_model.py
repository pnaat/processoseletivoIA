import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Carregamento do dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Pré-processamento: Normalização (0 a 1) e redimensionamento (adicionando o canal de cor)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 3. Construção do modelo CNN (Simples e otimizado para Edge AI)
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5), # Evita overfitting
    layers.Dense(10, activation="softmax"),
])

# 4. Compilação do modelo
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 5. Treinamento do modelo (Limitado a 5 épocas, focando em CPU e restrições de CI)
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# 6. Avaliação e salvamento do modelo
score = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final no teste: {score[1]:.4f}")

model.save("model.h5")
print("Modelo salvo com sucesso no formato Keras como 'model.h5'")
