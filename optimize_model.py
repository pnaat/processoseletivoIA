import tensorflow as tf
import os

# Carregando Modelo Treinado
# Bloco try-catch para tentar carregar o modelo, se ocorrer algum erro durante o processo uma exceção será disparada 
try:
    model = tf.keras.models.load_model('model.h5')
    print("Modelo 'model.h5' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar 'model.h5'. Verifique se o treinamento foi concluído.\nDetalhes: {e}")
    exit(1)


# Configurando conversor para TensorFlow Lite e aplicando Otimização com Dynamic Range Quantization
# Convertendo os pesos de float32 (32 bits) para int8 (8 bits), para reduzir o tamanho do arquivo sem perder muita precisão
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Salvando modelo otimizado
tflite_path = 'model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_quant_model)

# Informações sobre tamanhos do arquivo original, modificado e sobre a taxa de compressão
if os.path.exists('model.h5') and os.path.exists(tflite_path):
    size_h5 = os.path.getsize('model.h5') / 1024
    size_tflite = os.path.getsize(tflite_path) / 1024
    
    print(f"\n------ Relatório de Otimização ------")
    print(f"Tamanho Original (.h5):      {size_h5:.2f} KB")
    print(f"Tamanho Otimizado (.tflite): {size_tflite:.2f} KB")
    print(f"Taxa de Compressão:          {(1 - (size_tflite/size_h5))*100:.2f}%")

print("\nArquivo 'model.tflite' gerado com sucesso.")