import tensorflow as tf
import os

# 1. Carregamento do modelo treinado
model_path = "model.h5"
if not os.path.exists(model_path):
    print(f"Erro: O modelo {model_path} não foi encontrado. Execute train_model.py primeiro.")
    exit(1)

model = tf.keras.models.load_model(model_path)

# 2. Configuração do conversor TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Aplicação da técnica de otimização: Dynamic Range Quantization
# A flag Optimize.DEFAULT realiza a quantização de pesos para int8, reduzindo o tamanho
# enquanto as ativações permanecem em ponto flutuante durante a inferência.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Conversão para o formato .tflite
tflite_model = converter.convert()

# 5. Salvamento do modelo otimizado
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo convertido, otimizado e salvo como '{tflite_model_path}'")
