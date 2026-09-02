# Mini-Projeto 2 - Classificação de Dígitos MNIST

Este projeto cria um pipeline de análise preditiva para reconhecer dígitos manuscritos usando a base MNIST. A ideia é comparar modelos diferentes, avaliar os resultados com métricas e também testar como o modelo se comporta quando recebe dados que fogem do padrão usado no treino.

## Objetivo do sistema

O sistema recebe imagens de dígitos manuscritos no formato do MNIST e tenta classificar cada imagem como um número de 0 a 9. Além da classificação normal, o projeto também testa um cenário mais difícil: retirar algumas classes do treino e ver como o modelo reage quando encontra dígitos que ele nunca viu.

## O que foi desenvolvido

O projeto foi organizado nas fases pedidas no enunciado:

- Carregamento da base MNIST.
- Análise exploratória dos dados e visualização dos dígitos.
- Divisão dos dados em treino, validação e teste com estratificação.
- Normalização dos pixels para a escala de 0 a 1.
- Treinamento de 3 modelos: KNN, Random Forest e MLP.
- Comparação dos modelos usando acurácia, precisão, recall e F1-score.
- Matrizes de confusão para entender os erros.
- Teste OOD com classes ocultas, usando os dígitos 4 e 7.
- Pipeline para testar imagens próprias salvas na pasta `data/minhas_imagens`.

## Estrutura do projeto

```text
Mini-Projeto-2/
├── MiniProjeto2_MNIST.ipynb
├── mini_projeto_2.py
├── requirements.txt
├── README.md
├── PLANO_DE_PROJETO.md
├── script_video.txt
├── data/
│   └── minhas_imagens/
├── figuras/
├── models/
└── resultados/
```

## Tecnologias usadas

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Joblib
- Jupyter Notebook

## Como executar

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as bibliotecas:

```bash
pip install -r requirements.txt
```

Para rodar pelo VS Code:

```bash
python mini_projeto_2.py
```

Também é possível abrir o arquivo `MiniProjeto2_MNIST.ipynb` e executar as células em ordem.

## Observação sobre desempenho

O código vem com uma amostra rápida ativada para facilitar os testes no VS Code. Essa configuração está no começo do arquivo `mini_projeto_2.py`:

```python
USAR_AMOSTRA_RAPIDA = True
AMOSTRA_POR_CLASSE = 1000
```

Se quiser rodar com a base completa, basta trocar para:

```python
USAR_AMOSTRA_RAPIDA = False
```

## Imagens próprias

Para testar imagens feitas por mim, basta colocar arquivos `.png`, `.jpg`, `.jpeg` ou `.bmp` na pasta:

```text
data/minhas_imagens/
```

O código faz a conversão para escala de cinza, ajusta a imagem para 28x28 pixels, normaliza os valores e depois mostra a previsão do melhor modelo.

## Branches planejadas

- `main`: versão final do projeto.
- `develop`: branch usada para juntar as partes durante o desenvolvimento.
- `feature/eda-mnist`: carregamento e análise exploratória.
- `feature/preprocessamento`: divisão dos dados e normalização.
- `feature/modelos`: treinamento dos modelos.
- `feature/avaliacao`: métricas, matriz de confusão e comparação.
- `feature/robustez-ood`: teste com classes ocultas.
- `feature/imagens-proprias`: teste com imagens próprias.
- `feature/documentacao`: README, requirements e roteiro do vídeo.

O arquivo `PLANO_DE_PROJETO.md` também resume como eu organizei essas etapas e o objetivo de cada branch.

## Melhorias possíveis

Eu poderia melhorar o projeto treinando com a base completa em uma máquina com mais tempo de processamento, testando uma rede neural convolucional e criando uma interface visual para facilitar o envio das imagens. Também seria interessante salvar mais exemplos de previsões certas e erradas para entender melhor onde o modelo se confunde.
