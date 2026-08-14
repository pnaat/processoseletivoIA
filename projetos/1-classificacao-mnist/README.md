<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=header"/>

<h1 align="center">Classificação de Dígitos MNIST para Edge AI</h1>

<p align="center">
  Treinamento de uma rede neural convolucional, conversão para TensorFlow Lite
  e execução de inferências individuais em um artefato otimizado para Edge AI.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white"/>
  <img alt="Keras" src="https://img.shields.io/badge/Keras-3.3.x-D00000?logo=keras&logoColor=white"/>
  <img alt="Status" src="https://img.shields.io/badge/status-aguardando%20treinamento-yellow"/>
</p>

---

## Identificação

| Campo | Informação |
| --- | --- |
| Nome completo | **Mateus Alencar Ferreira** |
| GitHub | [@ferreiramateusalencar](https://github.com/ferreiramateusalencar) |
| Projeto escolhido | Classificação MNIST |

---

## 1. Resumo da arquitetura

O modelo recebe imagens em escala de cinza com formato **`(28, 28, 1)`**. Os
pixels são convertidos para `float32` e normalizados para o intervalo
**`[0, 1]`** antes da divisão dos dados.

A CNN contém três blocos convolucionais:

| Bloco | Filtros | Operações |
| ---: | ---: | --- |
| 1 | 16 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |
| 2 | 32 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |
| 3 | 64 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |

Após a extração de características, a saída é achatada por `Flatten` e passa
por uma camada densa com 64 neurônios e ativação ReLU. Em seguida,
**`Dropout(0.30)`** desativa aleatoriamente 30% das ativações durante o
treinamento. A camada final possui **10 neurônios com `softmax`**, um para cada
dígito de 0 a 9.

```text
Entrada (28×28×1)
  └─ Bloco Conv 16 filtros
      └─ Bloco Conv 32 filtros
          └─ Bloco Conv 64 filtros
              └─ Flatten → Dense(64, ReLU) → Dropout(30%)
                  └─ Dense(10, Softmax)
```

### Estratégia de treinamento

- **Treino:** 54.000 imagens do conjunto de treinamento do MNIST.
- **Validação:** 6.000 imagens separadas explicitamente antes do ajuste.
- **Máximo de épocas:** 12.
- **Batch size:** 128.
- **Otimizador:** Adam, com learning rate `0.001`.
- **Early stopping:** monitora `val_loss`, usa paciência de duas épocas e
  restaura os pesos da melhor época.
- **Dispositivo:** CPU, com GPU desabilitada pelo script.
- **Reprodutibilidade:** semente aleatória 42.

Foi utilizado **Dropout de 30%** para reduzir o risco de overfitting sem
comprometer excessivamente a capacidade de aprendizado. O **batch size de 128**
foi escolhido para diminuir o tempo de treinamento em CPU, mantendo
atualizações de gradiente suficientemente frequentes. O limite de 12 épocas,
com early stopping, impede treinamento desnecessário quando a perda de
validação deixa de melhorar.

---

## 2. Bibliotecas utilizadas

As versões abaixo estão fixadas para o ambiente de execução do projeto. A
versão efetivamente carregada de cada pacote será confirmada após a criação do
ambiente e antes do treinamento.

| Biblioteca | Versão | Utilização |
| --- | --- | --- |
| Python | 3.10 | Execução dos scripts e compatibilidade com a CI |
| TensorFlow | 2.16.1 | Treinamento e conversão para TensorFlow Lite |
| Keras | incluído no TensorFlow 2.16.1 | Construção e salvamento da CNN |
| NumPy | 1.26.4 | Preparação das imagens e interpretação das predições |

As versões reais podem ser verificadas no ambiente ativo com:

```bash
python --version
pip show tensorflow keras numpy
```

---

## 3. Técnica de otimização

Foi escolhida a **Quantização de Faixa Dinâmica** (*Dynamic Range
Quantization*), ativada no conversor TFLite com:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Durante a conversão, os pesos originalmente armazenados em `float32` são
geralmente quantizados para `int8`. As entradas e saídas permanecem em ponto
flutuante, portanto o script de inferência continua recebendo imagens
normalizadas em `float32`.

Essa técnica reduz o espaço de armazenamento e a memória necessária para os
pesos sem exigir um conjunto representativo de calibração. O arquivo resultante
é mais apropriado para dispositivos Edge, nos quais armazenamento, memória e
capacidade de processamento são limitados.

Após salvar `model.tflite`, o script cria um `tf.lite.Interpreter` e aloca seus
tensores. Essa verificação confirma que o artefato gerado pode ser interpretado
pelo runtime do TensorFlow Lite.

---

## 4. Resultados obtidos

> **Resultados pendentes:** o código corrigido ainda não foi treinado. Os
> campos abaixo serão substituídos exclusivamente pelos valores produzidos
> pelos novos artefatos; resultados antigos ou de outros repositórios não serão
> reutilizados.

| Resultado | Valor real |
| --- | ---: |
| Acurácia final de validação | Pendente de treinamento |
| Acurácia no validador oficial | Pendente de validação |
| Tamanho de `model.h5` | Pendente de geração |
| Tamanho de `model.tflite` | Pendente de conversão |
| Redução absoluta | Pendente de conversão |
| Redução percentual | Pendente de conversão |

Os tamanhos serão medidos diretamente em bytes e apresentados também em KB,
usando a mesma base de cálculo para os dois arquivos.

---

## 5. Comentários adicionais

### Dificuldade

O principal desafio é equilibrar tempo de treinamento em CPU, capacidade da
rede e regularização. Uma arquitetura muito pequena pode perder acurácia,
enquanto uma arquitetura profunda demais aumenta o custo sem benefício
proporcional para o MNIST.

### Decisão técnica

Foram usados três blocos com aumento progressivo de 16 para 64 filtros. Essa
configuração permite que as primeiras camadas aprendam formas simples e que as
camadas posteriores combinem essas formas em características mais abstratas,
mantendo o modelo compacto.

### Limitação

O modelo é treinado em imagens centralizadas, normalizadas e semelhantes às do
MNIST. Ele não deve ser considerado confiável para números fotografados em
papéis, placas ou ambientes reais sem novo treinamento e um conjunto de dados
representativo dessas condições.

### Aprendizado

O projeto demonstra que otimização para Edge AI não consiste apenas em reduzir
o tamanho do arquivo: é necessário verificar a métrica de validação, carregar o
artefato convertido e confirmar que ele ainda produz classificações coerentes.

---

## 6. Exemplo de inferência

Após gerar o modelo otimizado, a saída real será obtida com:

```bash
python run_inference.py
```

O script executa dez amostras e apresenta o formato:

```text
Inferência de 10 amostras com model.tflite
Amostra 01: predito=<resultado real> | real=<rótulo real> | confiança=<valor real>
...
Acertos na amostra: <total real>/10
```

> **Saída pendente:** este bloco será substituído pela saída integral do
> terminal depois do treinamento e da conversão. O comentário técnico será
> baseado nos acertos ou erros realmente observados.

---

<p align="center">
  <strong>Fluxo do projeto</strong><br/>
  Treinamento → Validação → Keras H5 → Quantização dinâmica → TensorFlow Lite → Inferência
</p>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=footer"/>
