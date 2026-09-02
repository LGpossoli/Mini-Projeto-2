# Plano de Projeto

Este arquivo resume como eu organizei o desenvolvimento do mini-projeto. A ideia foi separar o trabalho em etapas para ficar mais fácil de testar, explicar e versionar no GitHub.

## Branches usadas

- `main`: guarda a versão final entregue.
- `develop`: junta as etapas antes de mandar para a `main`.
- `feature/eda-mnist`: carregamento do MNIST e análise inicial das imagens.
- `feature/preprocessamento`: divisão estratificada e normalização dos pixels.
- `feature/modelos`: treinamento dos modelos KNN, Random Forest e MLP.
- `feature/avaliacao`: métricas, matrizes de confusão e comparação dos modelos.
- `feature/robustez-ood`: teste com classes ocultas e análise de falsa certeza.
- `feature/imagens-proprias`: leitura e tratamento das imagens manuscritas próprias.
- `feature/documentacao`: README, requirements e roteiro do vídeo.

## Etapa de EDA

Na primeira etapa eu carreguei a base MNIST, conferi as dimensões de `X` e `y`, analisei a distribuição dos dígitos e gerei uma grade visual com exemplos de 0 a 9. Essa parte ajuda a entender que cada imagem 28x28 é transformada em 784 valores de pixel.

## Etapa de pré-processamento

Depois da análise inicial, eu dividi os dados em treino, validação e teste usando estratificação. Fiz isso para manter a proporção dos dígitos em cada conjunto. Também normalizei os pixels para ficarem entre 0 e 1, porque assim os modelos trabalham com valores mais equilibrados.
