# Como contribuir

Obrigado pelo interesse em contribuir com este projeto de classificação MNIST
para Edge AI. Este documento descreve como propor mudanças de maneira
reproduzível e compatível com as regras do desafio.

## Antes de começar

Leia:

- o `README.md` da raiz, que contém as orientações gerais;
- o `README.md` de `projetos/1-classificacao-mnist/`, que descreve a
  implementação e os resultados;
- o `CODE_OF_CONDUCT.md`.

Abra uma issue antes de iniciar mudanças grandes na arquitetura, nas
dependências ou no formato dos artefatos. Isso permite discutir escopo e
impactos antes da implementação.

## Preparação do ambiente

O ambiente de referência usa Python 3.10. Crie um ambiente virtual isolado:

```bash
python -m venv .venv
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

No PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r projetos/1-classificacao-mnist/requirements.txt
```

Não instale dependências do projeto globalmente.

## Organização do repositório

A implementação deve permanecer em:

```text
projetos/1-classificacao-mnist/
├── train_model.py
├── optimize_model.py
├── run_inference.py
├── requirements.txt
├── model.h5
├── model.tflite
└── README.md
```

Não altere os workflows ou validadores oficiais para facilitar a aprovação de
uma contribuição. Mudanças nesses arquivos exigem justificativa técnica
explícita e revisão separada.

## Fluxo recomendado

1. Crie uma branch a partir de `main`.
2. Faça mudanças pequenas e relacionadas a um único objetivo.
3. Atualize testes e documentação quando necessário.
4. Execute as validações locais.
5. Abra um pull request explicando o problema, a solução e as evidências.

Exemplo:

```bash
git switch -c tipo/descricao-curta
```

Prefixos sugeridos: `feat/`, `fix/`, `docs/`, `refactor/` e `test/`.

## Validações locais

Confira a sintaxe:

```bash
python -m py_compile \
  projetos/1-classificacao-mnist/train_model.py \
  projetos/1-classificacao-mnist/optimize_model.py \
  projetos/1-classificacao-mnist/run_inference.py
```

Depois que os artefatos estiverem disponíveis, execute:

```bash
python .github/scripts/validate_1_mnist.py projetos/1-classificacao-mnist
cd projetos/1-classificacao-mnist
python run_inference.py
```

Não treine novamente apenas para modificar documentação. Quando um novo
treinamento for necessário, registre no pull request:

- ambiente e versões usadas;
- acurácia final de validação;
- tamanho de `model.h5` e `model.tflite`;
- redução percentual;
- saída real da inferência.

## Estilo de código

- Use nomes claros e código compatível com Python 3.10.
- Prefira funções pequenas e responsabilidades explícitas.
- Preserve a reprodutibilidade por meio de sementes aleatórias.
- Não inclua credenciais, datasets duplicados, caches ou ambientes virtuais.
- Fixe versões de novas dependências e justifique sua inclusão.
- Mantenha a documentação alinhada ao comportamento real do código.

## Commits

Escreva mensagens curtas e objetivas. O padrão Conventional Commits é
recomendado:

```text
feat: adiciona avaliação no conjunto de teste
fix: corrige normalização da entrada TFLite
docs: atualiza resultados reais da inferência
```

Evite misturar formatação, refatoração e mudança funcional no mesmo commit.

## Pull requests

Todo pull request deve informar:

- o objetivo da mudança;
- os arquivos afetados;
- como a mudança foi validada;
- métricas anteriores e novas, quando aplicável;
- limitações ou riscos conhecidos.

Ao contribuir, você concorda que sua participação seguirá o
`CODE_OF_CONDUCT.md` deste repositório.

