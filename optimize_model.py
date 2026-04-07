import tensorflow as tf
import os

# 1. Carregamento do modelo treinado
model_path = "model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Erro: O arquivo '{model_path}' não foi encontrado. Execute train_model.py primeiro.")

print("Carregando o modelo treinado...")
model = tf.keras.models.load_model(model_path)

# 2. Inicialização do conversor para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Aplicação de técnica de otimização (Dynamic Range Quantization)
# Isso converterá os pesos de float32 para int8, reduzindo o tamanho do modelo substancialmente
print("Aplicando otimização (Dynamic Range Quantization)...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Conversão do modelo
print("Convertendo para .tflite...")
tflite_quant_model = converter.convert()

# 5. Salvamento do modelo otimizado para Edge AI
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_quant_model)

# Exibindo métricas de otimização no terminal (Comparativo de tamanho)
h5_size = os.path.getsize(model_path) / 1024
tflite_size = os.path.getsize(tflite_model_path) / 1024
print(f"\n✅ Modelo convertido e salvo como '{tflite_model_path}'")
print(f"📉 Tamanho original (.h5): {h5_size:.2f} KB")
print(f"📉 Tamanho otimizado (.tflite): {tflite_size:.2f} KB")
print(f"🚀 Redução de espaço: {((h5_size - tflite_size) / h5_size) * 100:.1f}%")
