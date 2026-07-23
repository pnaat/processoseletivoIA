import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Carregar o modelo treinado
model = tf.keras.models.load_model("model.h5")

# Configurar o conversor
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aplicar a técnica de otimização
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# conversão
tflite_model = converter.convert()

# Salvar
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Otimização concluída. Arquivo 'model.tflite' gerado com sucesso.")
