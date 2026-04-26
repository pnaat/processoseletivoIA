import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path


# Configuração de Caminho
BASE_DIR = Path(__file__).resolve().parent
MODEL_H5_PATH = BASE_DIR / "model.h5"

# Carregando dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalização
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 28x28 -> 28x28x1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Construindo modelo CNN

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        
        # 1 Camada 
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 2 Camada
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dropout(0.3),         
        layers.Dense(32, activation="relu"),
        layers.Dense(10, activation="softmax"), 
    ]
)

print("\nResumo da Arquitetura: ")
model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n-> Treinando o modelo: ")
history = model.fit(
    x_train,
    y_train,
    batch_size=64, 
    epochs=5,
    validation_split=0.1,
    verbose=1,
)

# Avaliando o modelo e extraindo métricas de desempenho
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Implementação manual do F1-Score sem dependências externas para evitar falhas de bibliotecas
test_probabilities = model.predict(x_test, verbose=0)
test_predictions = np.argmax(test_probabilities, axis=1)

f1_scores = []
for i in range(10):
    # tp - true positive, fp - false positive, fn - false negative
    tp = np.sum((test_predictions == i) & (y_test == i))
    fp = np.sum((test_predictions == i) & (y_test != i))
    fn = np.sum((test_predictions != i) & (y_test == i))
    
    # p - precision, r - recall 
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

f1_macro = float(np.mean(f1_scores))

print(f"\n----- Resultados Finais -----")
print(f"Loss no teste: {test_loss:.4f}")
print(f"Acurácia: {test_accuracy * 100:.2f}%")
print(f"F1-Score Macro: {f1_macro:.4f}")

print("\n-> Gerando arquivo do modelo. ")
model.save(MODEL_H5_PATH)
print(f"Modelo salvo com sucesso em: {MODEL_H5_PATH}")































