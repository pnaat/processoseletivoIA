# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Taylan Mayckon Oliveira Araujo**

### 1️⃣ Resumo da Abordagem

<!-- Descreva os hiperparâmetros de fine-tuning utilizados (épocas, tamanho de
imagem, batch size) e quaisquer ajustes feitos para lidar com o desbalanceamento
de classes, se houver. -->
O fine-tuning foi realizado a partir do modelo pré-treinado YOLO11n, utilizando o dataset de detecção de máscaras fornecido pelo processo seletivo, já convertido para o formato YOLO. Antes do treinamento, conduziu-se uma análise exploratória para verificar a distribuição das classes e a consistência das anotações.

O conjunto de dados é composto por 683 imagens de treinamento e 170 de validação, distribuídas em três classes:

| Classe | Quantidade de objetos | Distribuição | 
|----------|----------|----------|
| with_mask   | 3232 | 79,37% | 
| without_mask | 717 | 17,61% | 
| mask_weared_incorrect | 123 | 3,02% |

Observa-se um desbalanceamento acentuado: a classe mask_weared_incorrect possui cerca de 26 vezes menos exemplos que with_mask. Uma inspeção visual de amostras das anotações, realizada com um script auxiliar, não indicou problemas de posicionamento das bounding boxes ou de conversão do formato original.

Os hiperparâmetros utilizados no treinamento foram:

| Parâmetro | Valor |
|----------|----------|
| Modelo base | YOLO11n |
| Pesos iniciais | Pré-treinados (pretrained=True) |
| Épocas | 20 |
| Tamanho da imagem (imgsz) | 640x640 |
| Batch size | 8 |
| Dispositivo | CPU |
| Otimizador | Auto (seleção automática da Ultralytics) |
| Seed | 42 |
| Workers | 0 |
| Early stopping (patience) | 10 |

A definição de 20 épocas buscou equilibrar tempo de treinamento e convergência, considerando que o modelo parte de pesos pré-treinados e que o dataset é de tamanho reduzido. A resolução de entrada foi mantida em 640x640 pixels em função de uma constatação da análise exploratória: as bounding boxes ocupam, em média, apenas 1,64% da área das imagens, o que caracteriza um cenário de detecção de objetos pequenos, reduzir a resolução tenderia a comprometer ainda mais a informação disponível sobre os rostos. O batch size 8 foi adotado como valor conservador para viabilizar o treinamento em CPU sem comprometer a estabilidade.

Não foram aplicadas técnicas específicas para mitigar o desbalanceamento entre classes, como oversampling ou ponderação de perda. Optou-se por manter o pipeline padrão da Ultralytics e avaliar posteriormente o impacto do desbalanceamento nas métricas por classe, já que era esperado desempenho inferior para mask_weared_incorrect dada sua baixa representatividade. O parâmetro patience=10 habilitou o early stopping, mecanismo que não chegou a ser acionado, uma vez que o treinamento apresentou melhora contínua ao longo das 20 épocas executadas.


### 2️⃣ Bibliotecas Utilizadas

<!-- Liste as principais bibliotecas utilizadas, preferencialmente com suas versões. -->

| Biblioteca | Versão | Finalidade |
|----------|----------|----------|
| Python | 3.10.3 | Linguagem utilizada no desenvolvimento |
| Ultralytics | 8.4.26 | Treinamento, validação, exportação e inferência do modelo YOLO11n |
| PyTorch | 2.13.0 | Backend utilizado pela Ultralytics durante o treinamento |
| TensorFlow | 2.19.0 | Execução da inferência com o modelo TensorFlow Lite|
| TensorFlow Lite | 2.19.0 | Execução do modelo model.tflite |

As bibliotecas os e shutil, empregadas na manipulação de arquivos e diretórios, integram a biblioteca padrão do Python e não possuem versionamento independente.


### 3️⃣ Técnica de Otimização do Modelo

<!-- Explique o processo de exportação para TFLite realizado em `optimize_model.py`. -->
O modelo treinado (model.pt) foi convertido para o formato TensorFlow Lite pelo script optimize_model.py, por meio da função model.export(format="tflite") da biblioteca Ultralytics.

Nas versões recentes da Ultralytics, o processo de exportação passou a gerar uma pasta intermediária (model_saved_model) contendo múltiplos artefatos da conversão, entre eles as versões Float32 e Float16 do modelo TensorFlow Lite, além de arquivos auxiliares como o modelo ONNX e dados de calibração. Para adequar essa saída ao formato esperado pelo desafio, o script localiza automaticamente o arquivo model_float16.tflite, copia-o para a raiz do projeto e o renomeia para model.tflite.

A escolha pela versão Float16 justifica-se pela redução no tamanho do arquivo e no consumo de memória em relação à versão Float32, sem perda perceptível de desempenho na inferência, estratégia comum em aplicações de Edge AI, nas quais os recursos computacionais são limitados.

### 4️⃣ Resultados Obtidos

<!-- Informe o mAP50 (e, se possível, o mAP50-95) obtido na validação, por classe se
possível, e o tamanho dos arquivos `model.pt` e `model.tflite`. -->

As métricas de validação obtidas ao final do treinamento foram:

| Métrica | Valor |
|----------|----------|
| Precision | 0,740 |
| Recall | 0,700 |
| mAP50 | 0,748 |
| mAP50-95 | 0,527 |

O desempenho por classe é apresentado a seguir:

| Classe | Precision | Recall | mAP50 | mAP50-95 |
|----------|----------|----------|----------|----------|
| with_mask | 0,908 | 0,939 | 0,966 | 0,685 |
| without_mask | 0,722 | 0,705 | 0,766 | 0,500 |
| mask_weared_incorrect | 0,589 | 0,454 | 0,513 | 0,395 |

A classe with_mask apresentou o melhor desempenho, seguida por without_mask. O resultado inferior de mask_weared_incorrect era esperado, dado que essa classe representa apenas cerca de 3% dos objetos anotados no conjunto de dados.

Os tamanhos dos modelos gerados foram:

| Arquivo | Tamanho |
|----------|----------|
| model.pt | 5,20 MB |
| model.tflite | 5,11 MB |

A conversão para TensorFlow Lite resultou em uma redução discreta de tamanho, sem impacto perceptível sobre o desempenho do modelo.


### 5️⃣ Comentários Adicionais (Opcional)

<!-- Dificuldades encontradas, decisões técnicas importantes, limitações do modelo
(ex: desempenho na classe minoritária), aprendizados durante o desafio. -->

A principal limitação observada ao longo do desenvolvimento decorre do desbalanceamento do conjunto de dados: a classe with_mask responde por cerca de 79% dos objetos anotados, contra apenas 3% de mask_weared_incorrect. Esse desequilíbrio se refletiu diretamente nas métricas por classe, mas ainda assim o modelo foi capaz de aprender padrões distintos para as três categorias, o que sugere que o pipeline de fine-tuning funcionou de maneira consistente mesmo diante de dados escassos para a classe minoritária.

Um segundo ponto relevante foi a exportação para TensorFlow Lite. Alterações recentes na biblioteca Ultralytics passaram a gerar múltiplos artefatos intermediários durante a conversão, em vez de produzir diretamente um único arquivo .tflite. Isso exigiu a implementação de uma etapa adicional em optimize_model.py para localizar e selecionar automaticamente a versão Float16 entre os artefatos gerados.

De maneira geral, o projeto permitiu percorrer as etapas centrais de um pipeline de detecção de objetos voltado a Edge AI, da análise do conjunto de dados ao fine-tuning, exportação e validação do modelo em formato otimizado.



### 6️⃣ Exemplo de Inferência

<!-- Cole a saída do terminal ao rodar `run_inference.py` (número de detecções por
imagem), e comente brevemente sobre o que observou ao abrir as imagens
anotadas em `runs/detect/inferencia_exemplos/predicoes/` — por exemplo, se as
caixas ficaram bem localizadas, se houve confusão entre classes, ou se a
classe minoritária (`mask_weared_incorrect`) teve desempenho visivelmente pior. -->

O script run_inference.py foi executado sobre cinco imagens do conjunto de validação, utilizando exclusivamente o modelo model.tflite. A saída obtida foi:

Rodando inferência em 5 amostras usando model.tflite:

| Imagem | Detecções | Detalhes |
|----------|----------|----------|
| maksssksksss105.jpg | 9 | [9x with_mask] |
| maksssksksss107.jpg  | 1 | [1x with_mask] |
| maksssksksss11.jpg | 24 | [1x mask_weared_incorrect, 21x with_mask, 2x without_mask] |
| maksssksksss113.jpg  | 4 | [3x with_mask, 1x without_mask] |
| maksssksksss12.jpg  | 16 | [12x with_mask, 4x without_mask] |
| TOTAL | 54 |  |

Imagens anotadas salvas em:
runs/detect/inferencia_exemplos/predicoes/

As imagens anotadas foram então comparadas visualmente com as imagens originais. As bounding boxes apresentaram boa localização sobre os rostos detectados, inclusive em imagens com grande número de pessoas, nas quais o modelo conseguiu identificar corretamente diversos rostos simultaneamente.

Não foram observadas confusões evidentes entre classes nas amostras analisadas. Registrou-se uma detecção da classe mask_weared_incorrect, com confiança de aproximadamente 0,93 e bounding box corretamente posicionada. Ainda que essa classe apresente o desempenho mais baixo nas métricas agregadas de validação, resultado do forte desbalanceamento do conjunto de dados, essa limitação não se manifestou de forma clara nas imagens utilizadas para inferência.

Em síntese, a inspeção visual corroborou as métricas de validação, indicando que o modelo exportado para TensorFlow Lite manteve comportamento compatível com o modelo original treinado em PyTorch.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
