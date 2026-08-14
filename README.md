<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=header" alt="Cabeçalho decorativo"/>

<h1 align="center">
  <img src="https://fit-tecnologia.org.br/ava/pluginfile.php/1/theme_moove/logo/1784901257/pnaat-positivo.png" width="300px" alt="PNAAT"/>
  <br/>
  Desafio Técnico — Edge AI com Visão Computacional
</h1>

<p align="center">
  Classificação de dígitos manuscritos do MNIST com uma Rede Neural
  Convolucional, conversão para TensorFlow Lite e inferência otimizada para
  dispositivos Edge.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white"/>
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white"/>
  <img alt="Status" src="https://img.shields.io/badge/status-concluído-brightgreen"/>
  <a href="https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/wiki">
    <img alt="Wiki" src="https://img.shields.io/badge/documentação-Wiki-181717?logo=github"/>
  </a>
  <a href="https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/actions/workflows/ci.yml">
    <img alt="GitHub Actions" src="https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/actions/workflows/ci.yml/badge.svg"/>
  </a>
</p>

---

## 👤 Identificação

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/86336670?v=4" width="100px" alt="Mateus Alencar Ferreira"/>
      <br/>
      <strong>Mateus Alencar Ferreira</strong>
      <br/>
      <a href="https://github.com/ferreiramateusalencar">@ferreiramateusalencar</a>
      <br/><br/>
      <a href="https://www.linkedin.com/in/mateus-alencar-ferreira/">🌐 LinkedIn</a>
    </td>
  </tr>
</table>

| Campo | Informação |
| --- | --- |
| Processo seletivo | PNAAT — Intensivo Maker |
| Projeto escolhido | Projeto 1 — Classificação MNIST |
| Tarefa | Classificação de dígitos manuscritos de 0 a 9 |
| Modelo treinado | `model.h5` |
| Modelo otimizado | `model.tflite` |
| Execução automatizada | GitHub Actions |

---

## 📌 Sobre o projeto

Este projeto implementa o fluxo completo de uma aplicação de Visão
Computacional voltada para Edge AI:

```text
MNIST
  ↓
Preparação e normalização dos dados
  ↓
Treinamento e validação da CNN
  ↓
Salvamento do modelo Keras — model.h5
  ↓
Quantização de faixa dinâmica
  ↓
Modelo TensorFlow Lite — model.tflite
  ↓
Inferência individual em dispositivo Edge
```

O objetivo não é apenas classificar dígitos, mas demonstrar um pipeline
reproduzível de treinamento, validação, otimização e inferência com um artefato
adequado a ambientes com recursos computacionais limitados.

---

## 🧭 Navegação

- [Projeto MNIST](projetos/1-classificacao-mnist/)
- [Relatório técnico do projeto](projetos/1-classificacao-mnist/README.md)
- [Wiki completa](https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/wiki)
- [GitHub Actions](https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/actions)
- [Repositório-base](https://github.com/pnaat/processoseletivoIA)

---

## 🧠 Arquitetura do modelo

A CNN recebe imagens em escala de cinza com formato `(28, 28, 1)`. Os pixels
são convertidos para `float32` e normalizados do intervalo `[0, 255]` para
`[0, 1]`.

### Blocos convolucionais

| Bloco | Filtros | Estrutura |
| ---: | ---: | --- |
| 1 | 16 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |
| 2 | 32 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |
| 3 | 64 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |

Depois da extração de características, o modelo utiliza:

- `Flatten`;
- `Dense(64, ReLU)`;
- `Dropout(0.30)`;
- `Dense(10, Softmax)`.

```text
Entrada (28×28×1)
    ↓
Conv2D(16) → BatchNorm → ReLU → MaxPooling
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPooling
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPooling
    ↓
Flatten → Dense(64, ReLU) → Dropout(30%)
    ↓
Dense(10, Softmax)
```

O `Dropout` reduz o risco de overfitting. O aumento progressivo de filtros
permite que a rede aprenda desde traços simples até combinações mais abstratas
dos formatos dos dígitos.

---

## 📚 Dados e treinamento

O MNIST fornece 60.000 imagens no conjunto original de treinamento. O projeto
faz uma divisão explícita:

| Subconjunto | Quantidade | Finalidade |
| --- | ---: | --- |
| Treinamento | 54.000 | atualização dos pesos |
| Validação | 6.000 | acompanhamento de perda e acurácia |
| Teste | 10.000 | inferência e validação externa |

### Hiperparâmetros

| Parâmetro | Configuração |
| --- | --- |
| Otimizador | Adam |
| Learning rate | `0.001` |
| Função de perda | `sparse_categorical_crossentropy` |
| Máximo de épocas | 12 |
| Batch size | 128 |
| Early stopping | `val_loss`, paciência de 2 épocas |
| Melhor peso | restaurado automaticamente |
| Dispositivo | CPU |
| Semente aleatória | 42 |

O limite de 12 épocas, combinado com `EarlyStopping`, evita processamento
desnecessário quando a perda de validação deixa de melhorar. O batch size de
128 equilibra velocidade e frequência de atualização dos gradientes durante o
treinamento em CPU.

---

## ⚡ Otimização para Edge AI

O arquivo `optimize_model.py` converte o modelo Keras para TensorFlow Lite e
aplica **Quantização de Faixa Dinâmica**:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Essa técnica reduz principalmente a representação dos pesos, diminuindo o
tamanho do arquivo e a memória necessária para armazená-lo. Ela não exige um
dataset representativo de calibração e mantém entradas e saídas compatíveis
com o fluxo em ponto flutuante utilizado pelo projeto.

Depois da conversão, o `model.tflite` é carregado por
`tf.lite.Interpreter` e seus tensores são alocados, confirmando que o
artefato pode ser interpretado corretamente.

---

## 📊 Resultados

| Métrica | Resultado |
| --- | ---: |
| Acurácia registrada na execução de teste | **98,94%** |
| Tamanho de `model.h5` | **1.165.592 bytes — 1.138,3 KB** |
| Tamanho de `model.tflite` | **104.200 bytes — 101,8 KB** |
| Redução absoluta | **1.061.392 bytes** |
| Redução percentual | **91,1%** |

> As métricas devem ser associadas à versão correspondente dos artefatos. Caso
> os scripts de treinamento ou conversão sejam alterados, regenere os modelos
> e atualize os resultados com a nova execução.

---

## 🔎 Inferência com TensorFlow Lite

O `run_inference.py` carrega especificamente `model.tflite` e executa
inferência em dez imagens do conjunto de teste.

Para cada amostra, o terminal apresenta:

- dígito predito;
- dígito real;
- maior probabilidade produzida pelo modelo;
- quantidade total de acertos na amostra.

Essa etapa confirma que o modelo otimizado não é apenas um arquivo menor: ele
continua funcional e capaz de produzir classificações individuais.

---

## 📂 Estrutura do repositório

```text
PNAAT-processoseletivoIA/
├── .github/
│   ├── scripts/
│   │   ├── validate_common.py
│   │   └── validate_1_mnist.py
│   └── workflows/
│       └── ci.yml
├── projetos/
│   └── 1-classificacao-mnist/
│       ├── train_model.py
│       ├── optimize_model.py
│       ├── run_inference.py
│       ├── requirements.txt
│       ├── model.h5
│       ├── model.tflite
│       └── README.md
├── LICENSE
└── README.md
```

| Arquivo | Responsabilidade |
| --- | --- |
| `train_model.py` | preparação dos dados, construção e treinamento da CNN |
| `optimize_model.py` | conversão e otimização para TensorFlow Lite |
| `run_inference.py` | inferência individual com o modelo otimizado |
| `model.h5` | modelo Keras treinado |
| `model.tflite` | artefato otimizado para Edge AI |
| `ci.yml` | validação automatizada pelo GitHub Actions |

---

## ▶️ Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA.git
cd PNAAT-processoseletivoIA
```

### 2. Crie um ambiente virtual

#### Linux ou macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r projetos/1-classificacao-mnist/requirements.txt
```

### 4. Execute o pipeline

```bash
cd projetos/1-classificacao-mnist
python train_model.py
python optimize_model.py
python run_inference.py
```

---

## ✅ Validação automatizada

O workflow é executado em `push` e `pull_request` para a branch `main`.

Ele:

1. detecta qual projeto foi escolhido;
2. confirma a presença dos arquivos obrigatórios;
3. instala as dependências;
4. carrega e avalia os artefatos versionados;
5. executa treinamento e conversão;
6. testa a inferência com `model.tflite`.

O validador do MNIST usa 300 amostras do conjunto de teste e exige:

| Verificação | Critério |
| --- | ---: |
| Formato de entrada | `(28, 28, 1)` |
| Acurácia mínima de `model.h5` | 85% |
| Acurácia mínima de `model.tflite` | 75% |

Para executar o validador manualmente a partir da raiz:

```bash
PYTHONPATH=.github/scripts python .github/scripts/validate_1_mnist.py projetos/1-classificacao-mnist
```

---

## 📚 Documentação

A documentação detalhada está disponível na
[Wiki do repositório](https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/wiki).

Ela apresenta:

- visão geral e fluxo da solução;
- instalação e execução;
- arquitetura da CNN;
- preparação dos dados e treinamento;
- otimização e inferência;
- pipeline de validação;
- resultados e limitações;
- solução de problemas;
- referências e licenciamento.

O [README técnico do projeto](projetos/1-classificacao-mnist/README.md) mantém as
seções exigidas pela avaliação.

---

## ⚠️ Limitações

- O modelo foi desenvolvido para imagens padronizadas do MNIST.
- Não foi validado para números fotografados em ambientes reais.
- A confiança produzida pelo Softmax não garante, isoladamente, que a previsão
  esteja correta.
- Alterações na arquitetura exigem novo treinamento, conversão e validação.
- O desempenho em dispositivos Edge reais depende do hardware e do runtime.

---

## 📄 Licenciamento

Este repositório é um fork de
[pnaat/processoseletivoIA](https://github.com/pnaat/processoseletivoIA).

O conteúdo proveniente do repositório-base permanece sujeito aos direitos e
termos de seus respectivos titulares. As contribuições originais de Mateus
Alencar Ferreira neste fork são disponibilizadas sob os termos indicados no
arquivo [LICENSE](LICENSE).

---

## 👨‍💻 Autor

**Mateus Alencar Ferreira**

- GitHub: [@ferreiramateusalencar](https://github.com/ferreiramateusalencar)
- LinkedIn: [mateus-alencar-ferreira](https://www.linkedin.com/in/mateus-alencar-ferreira/)

<p align="center">
  <strong>Projeto desenvolvido para o Processo Seletivo PNAAT — Intensivo Maker</strong>
</p>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=footer" alt="Rodapé decorativo"/>
