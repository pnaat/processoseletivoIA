# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Erick Felipe

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada em train_model.py é uma Rede Neural Convolucional (CNN) sequencial leve. Ela é composta por 3 blocos convolucionais encadeados, onde cada bloco contém uma camada Conv2D para extração de características, seguida de BatchNormalization para estabilizar o aprendizado e MaxPooling2D para redução de dimensionalidade. Após a extração, os dados são achatados (Flatten) e passam por uma camada de Dropout (0.5) atuando como regularizador tático para evitar overfitting. A saída é uma camada Dense com ativação softmax para as 10 classes.
A estratégia de validação utilizou um validation_split de 10% sobre o conjunto de treino, acoplado a um callback de EarlyStopping monitorando a val_loss (com paciência de 3 épocas e restauração dos melhores pesos) para garantir eficiência de processamento em CPU.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow / Keras: Versão >= 2.12
NumPy: Instalado via resolução de dependências do ambiente
OS: Biblioteca nativa padrão do Python

### 3️⃣ Técnica de Otimização do Modelo

A otimização em optimize_model.py foi realizada através de Dynamic Range Quantization utilizando o tf.lite.Optimize.DEFAULT. Esta técnica reduz a precisão dos pesos da rede de ponto flutuante de 32 bits (Float32) para inteiros de 8 bits (Int8). Isso comprime drasticamente o tamanho físico do modelo e acelera a inferência em processadores limitados (Edge AI), mantendo a acurácia praticamente inalterada.

### 4️⃣ Resultados Obtidos

Acurácia de Validação: 99.08%

Tamanho do arquivo model.h5: 1,3M

Tamanho do arquivo model.tflite: 117K

### 5️⃣ Comentários Adicionais (Opcional)

A decisão técnica de limitar a rede a 3 blocos convolucionais provou-se altamente eficiente para o ambiente restrito do edital (treinamento apenas em CPU). O modelo atingiu convergência ideal e ativou o EarlyStopping rapidamente. A quantização foi extremamente bem-sucedida, reduzindo o tamanho do modelo em cerca de 90% (de 1,3M para 117K) sem perda aparente de precisão nos testes práticos.

### 6️⃣ Exemplo de Inferência

Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Como observado nas amostras acima, o modelo TFLite manteve sua integridade analítica após a quantização Int8, acertando com precisão o dígito 7 na Amostra 1 e o dígito 4 na Amostra 5, comprovando que a drástica redução de tamanho não penalizou a inferência na prática.
