from pathlib import Path
from time import perf_counter
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGENS_PROPRIAS_DIR = DATA_DIR / "minhas_imagens"
FIGURAS_DIR = BASE_DIR / "figuras"
RESULTADOS_DIR = BASE_DIR / "resultados"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
CLASSES = list(range(10))

# Usei uma amostra para o projeto rodar mais rápido no VS Code.
# Se quiser treinar com a base completa, troque USAR_AMOSTRA_RAPIDA para False.
USAR_AMOSTRA_RAPIDA = True
AMOSTRA_POR_CLASSE = 1000


def preparar_pastas():
    """Cria as pastas usadas para salvar gráficos, resultados e modelo."""
    for pasta in [DATA_DIR, IMAGENS_PROPRIAS_DIR, FIGURAS_DIR, RESULTADOS_DIR, MODELS_DIR]:
        pasta.mkdir(parents=True, exist_ok=True)


def carregar_mnist():
    """Baixa o MNIST pelo scikit-learn e devolve X e y."""
    print("Baixando o dataset MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    print("Finalizou o processo de baixar o dataset MNIST!")
    print(f"Formato de X: {X.shape}")
    print(f"Formato de y: {y.shape}")
    return X, y


def criar_amostra_estratificada(X, y, n_por_classe=1000):
    """Mantém a mesma quantidade aproximada de imagens por dígito."""
    rng = np.random.default_rng(RANDOM_STATE)
    indices = []

    for classe in CLASSES:
        indices_classe = np.where(y == classe)[0]
        qtd = min(n_por_classe, len(indices_classe))
        indices.extend(rng.choice(indices_classe, size=qtd, replace=False))

    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def mostrar_distribuicao_classes(y, nome_arquivo="distribuicao_classes.png"):
    """Mostra se os dígitos estão bem distribuídos na base."""
    distribuicao = pd.Series(y).value_counts().sort_index()
    print("\nDistribuição das classes:")
    print(distribuicao)

    plt.figure(figsize=(9, 4))
    sns.barplot(
        x=distribuicao.index,
        y=distribuicao.values,
        hue=distribuicao.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Distribuição dos dígitos no MNIST")
    plt.xlabel("Dígito")
    plt.ylabel("Quantidade de imagens")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / nome_arquivo, dpi=150)
    plt.show()
    return distribuicao


def mostrar_grade_digitos(X, y, nome_arquivo="grade_digitos_mnist.png"):
    """Exibe um exemplo de imagem para cada dígito de 0 a 9."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.ravel()

    for digito in CLASSES:
        indice = np.where(y == digito)[0][0]
        imagem = X[indice].reshape(28, 28)
        axes[digito].imshow(imagem, cmap="gray")
        axes[digito].set_title(f"Rótulo: {digito}")
        axes[digito].axis("off")

    plt.suptitle("Exemplos de dígitos manuscritos do MNIST")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / nome_arquivo, dpi=150)
    plt.show()


def dividir_e_normalizar(X, y):
    """Divide os dados com estratificação e normaliza os pixels para 0 até 1."""
    X_treino_val, X_teste, y_treino_val, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_treino_val,
        y_treino_val,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=y_treino_val,
    )

    X_treino = X_treino / 255.0
    X_validacao = X_validacao / 255.0
    X_teste = X_teste / 255.0

    print("\nTamanhos após a divisão:")
    print(f"Treino: {X_treino.shape[0]} imagens")
    print(f"Validação: {X_validacao.shape[0]} imagens")
    print(f"Teste: {X_teste.shape[0]} imagens")
    return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste


def treinar_modelos(X_treino, y_treino, X_validacao, y_validacao):
    """Treina 3 modelos e escolhe a melhor configuração de cada um pela validação."""
    configuracoes = {
        "KNN": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "distance"},
            {"n_neighbors": 7, "weights": "distance"},
        ],
        "Random Forest": [
            {"n_estimators": 80, "max_depth": 12},
            {"n_estimators": 120, "max_depth": 18},
            {"n_estimators": 120, "max_depth": None},
        ],
        "MLP": [
            {"hidden_layer_sizes": (64,), "alpha": 0.0001, "learning_rate_init": 0.001},
            {"hidden_layer_sizes": (128,), "alpha": 0.0001, "learning_rate_init": 0.001},
            {"hidden_layer_sizes": (128, 64), "alpha": 0.0005, "learning_rate_init": 0.001},
        ],
    }

    melhores_modelos = {}
    historico = []

    for nome_modelo, lista_parametros in configuracoes.items():
        print(f"\nTreinando {nome_modelo}...")
        melhor_acuracia = -1
        melhor_modelo = None
        melhor_parametros = None

        for parametros in lista_parametros:
            inicio = perf_counter()

            if nome_modelo == "KNN":
                modelo = KNeighborsClassifier(**parametros)
            elif nome_modelo == "Random Forest":
                modelo = RandomForestClassifier(
                    **parametros,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            else:
                modelo = MLPClassifier(
                    **parametros,
                    max_iter=30,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                )

            modelo.fit(X_treino, y_treino)
            tempo = perf_counter() - inicio

            pred_treino = modelo.predict(X_treino)
            pred_validacao = modelo.predict(X_validacao)
            acuracia_treino = accuracy_score(y_treino, pred_treino)
            acuracia_validacao = accuracy_score(y_validacao, pred_validacao)

            historico.append(
                {
                    "modelo": nome_modelo,
                    "parametros": str(parametros),
                    "acuracia_treino_%": acuracia_treino * 100,
                    "acuracia_validacao_%": acuracia_validacao * 100,
                    "tempo_treino_segundos": tempo,
                }
            )

            print(
                f"{nome_modelo} {parametros} | "
                f"treino: {acuracia_treino * 100:.2f}% | "
                f"validação: {acuracia_validacao * 100:.2f}% | "
                f"tempo: {tempo:.1f}s"
            )

            if acuracia_validacao > melhor_acuracia:
                melhor_acuracia = acuracia_validacao
                melhor_modelo = modelo
                melhor_parametros = parametros

        melhores_modelos[nome_modelo] = {
            "modelo": melhor_modelo,
            "parametros": melhor_parametros,
            "acuracia_validacao": melhor_acuracia,
        }

    historico_df = pd.DataFrame(historico)
    historico_df.to_csv(RESULTADOS_DIR / "historico_treinamento.csv", index=False)
    return melhores_modelos, historico_df


def avaliar_modelos(melhores_modelos, X_teste, y_teste):
    """Avalia os melhores modelos no teste e salva as métricas comparativas."""
    linhas = []
    predicoes = {}

    for nome_modelo, info in melhores_modelos.items():
        modelo = info["modelo"]
        inicio = perf_counter()
        y_pred = modelo.predict(X_teste)
        tempo_predicao = perf_counter() - inicio
        predicoes[nome_modelo] = y_pred

        linhas.append(
            {
                "modelo": nome_modelo,
                "parametros": str(info["parametros"]),
                "acuracia_%": accuracy_score(y_teste, y_pred) * 100,
                "precisao_ponderada_%": precision_score(
                    y_teste, y_pred, average="weighted", zero_division=0
                )
                * 100,
                "recall_ponderado_%": recall_score(
                    y_teste, y_pred, average="weighted", zero_division=0
                )
                * 100,
                "f1_ponderado_%": f1_score(y_teste, y_pred, average="weighted", zero_division=0)
                * 100,
                "tempo_predicao_segundos": tempo_predicao,
            }
        )

        relatorio = classification_report(y_teste, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(relatorio).transpose().to_csv(
            RESULTADOS_DIR / f"classification_report_{normalizar_nome(nome_modelo)}.csv"
        )

        plotar_matriz_confusao(
            y_teste,
            y_pred,
            titulo=f"Matriz de confusão - {nome_modelo}",
            nome_arquivo=f"matriz_confusao_{normalizar_nome(nome_modelo)}.png",
        )

    comparativo = pd.DataFrame(linhas).sort_values("acuracia_%", ascending=False)
    comparativo.to_csv(RESULTADOS_DIR / "comparativo_modelos.csv", index=False)
    print("\nTabela comparativa final:")
    print(comparativo.round(2))
    return comparativo, predicoes


def plotar_matriz_confusao(y_real, y_pred, titulo, nome_arquivo, labels=CLASSES):
    """Cria o heatmap da matriz de confusão."""
    matriz = confusion_matrix(y_real, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(titulo)
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / nome_arquivo, dpi=150)
    plt.show()
    return matriz


def identificar_maior_confusao(y_real, y_pred):
    """Mostra qual par de dígitos teve mais erro na matriz de confusão."""
    matriz = confusion_matrix(y_real, y_pred, labels=CLASSES)
    matriz_sem_diagonal = matriz.copy()
    np.fill_diagonal(matriz_sem_diagonal, 0)
    linha, coluna = np.unravel_index(np.argmax(matriz_sem_diagonal), matriz_sem_diagonal.shape)
    total = matriz_sem_diagonal[linha, coluna]
    return linha, coluna, total


def treinar_teste_ood(X_treino, y_treino, X_teste, y_teste, melhores_modelos, classes_ocultas=(4, 7)):
    """Treina o melhor tipo de modelo sem duas classes e testa só nessas classes ocultas."""
    melhor_nome = max(melhores_modelos, key=lambda nome: melhores_modelos[nome]["acuracia_validacao"])
    parametros = melhores_modelos[melhor_nome]["parametros"]

    mascara_treino = ~np.isin(y_treino, classes_ocultas)
    mascara_teste = np.isin(y_teste, classes_ocultas)

    X_treino_ood = X_treino[mascara_treino]
    y_treino_ood = y_treino[mascara_treino]
    X_teste_ood = X_teste[mascara_teste]
    y_teste_ood = y_teste[mascara_teste]

    if melhor_nome == "KNN":
        modelo_ood = KNeighborsClassifier(**parametros)
    elif melhor_nome == "Random Forest":
        modelo_ood = RandomForestClassifier(
            **parametros,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        modelo_ood = MLPClassifier(
            **parametros,
            max_iter=30,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )

    print(f"\nTreinando teste OOD com o modelo {melhor_nome}, ocultando {classes_ocultas}.")
    modelo_ood.fit(X_treino_ood, y_treino_ood)
    y_pred_ood = modelo_ood.predict(X_teste_ood)

    matriz_ood = plotar_matriz_confusao(
        y_teste_ood,
        y_pred_ood,
        titulo=f"Teste OOD - classes ocultas {classes_ocultas}",
        nome_arquivo="matriz_confusao_ood.png",
        labels=CLASSES,
    )

    contagem_prevista = pd.Series(y_pred_ood).value_counts().sort_index()
    contagem_prevista.to_csv(RESULTADOS_DIR / "previsoes_ood_classes_ocultas.csv")

    print("\nClasses que o modelo usou para tentar classificar os dígitos ocultos:")
    print(contagem_prevista)

    certeza_media = None
    if hasattr(modelo_ood, "predict_proba"):
        probabilidades = modelo_ood.predict_proba(X_teste_ood)
        certeza_media = probabilidades.max(axis=1).mean() * 100
        print(f"Certeza média nas previsões OOD: {certeza_media:.2f}%")

    return {
        "modelo": modelo_ood,
        "nome_modelo": melhor_nome,
        "classes_ocultas": classes_ocultas,
        "matriz": matriz_ood,
        "previsoes": y_pred_ood,
        "y_real": y_teste_ood,
        "certeza_media_%": certeza_media,
    }


def preprocessar_imagem_propria(caminho_imagem):
    """Converte uma imagem própria para o formato usado pelo MNIST."""
    imagem = Image.open(caminho_imagem).convert("L")
    imagem_array = np.array(imagem).astype(np.float32) / 255.0

    # Se o fundo estiver claro, inverto para ficar parecido com o MNIST.
    if imagem_array.mean() > 0.5:
        imagem = ImageOps.invert(imagem)
        imagem_array = np.array(imagem).astype(np.float32) / 255.0

    mascara = imagem_array > 0.15
    if mascara.any():
        linhas, colunas = np.where(mascara)
        topo, baixo = linhas.min(), linhas.max()
        esquerda, direita = colunas.min(), colunas.max()
        imagem = imagem.crop((esquerda, topo, direita + 1, baixo + 1))

    imagem.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    pos_x = (28 - imagem.width) // 2
    pos_y = (28 - imagem.height) // 2
    canvas.paste(imagem, (pos_x, pos_y))

    imagem_processada = np.array(canvas).astype(np.float32) / 255.0
    return imagem_processada


def testar_imagens_proprias(modelo, pasta_imagens=IMAGENS_PROPRIAS_DIR):
    """Faz previsões nas imagens próprias, se elas existirem na pasta."""
    extensoes = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    caminhos = []
    for extensao in extensoes:
        caminhos.extend(pasta_imagens.glob(extensao))

    if not caminhos:
        print(
            "\nNenhuma imagem própria foi encontrada. "
            "Coloque arquivos em data/minhas_imagens para testar essa parte."
        )
        return pd.DataFrame()

    resultados = []

    for caminho in caminhos:
        imagem_processada = preprocessar_imagem_propria(caminho)
        entrada = imagem_processada.reshape(1, -1)
        predicao = modelo.predict(entrada)[0]

        probabilidades = None
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada)[0]

        resultados.append({"arquivo": caminho.name, "predicao": int(predicao)})
        plotar_imagem_e_probabilidades(imagem_processada, probabilidades, predicao, caminho.stem)

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(RESULTADOS_DIR / "previsoes_imagens_proprias.csv", index=False)
    print("\nPrevisões das imagens próprias:")
    print(resultados_df)
    return resultados_df


def plotar_imagem_e_probabilidades(imagem_processada, probabilidades, predicao, nome_base):
    """Mostra a imagem processada junto com as probabilidades previstas."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(imagem_processada, cmap="gray")
    axes[0].set_title(f"Imagem processada\nPrevisão: {predicao}")
    axes[0].axis("off")

    if probabilidades is not None:
        classes_modelo = np.arange(len(probabilidades))
        axes[1].bar(classes_modelo, probabilidades * 100)
        axes[1].set_title("Probabilidades de saída")
        axes[1].set_xlabel("Classe")
        axes[1].set_ylabel("Probabilidade (%)")
        axes[1].set_xticks(classes_modelo)
    else:
        axes[1].text(0.5, 0.5, "Modelo sem probabilidades", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / f"imagem_propria_{normalizar_nome(nome_base)}.png", dpi=150)
    plt.show()


def salvar_melhor_modelo(melhores_modelos, comparativo):
    """Salva o melhor modelo para ser reutilizado em uma interface depois."""
    melhor_nome = comparativo.iloc[0]["modelo"]
    pacote = {
        "modelo": melhores_modelos[melhor_nome]["modelo"],
        "nome_modelo": melhor_nome,
        "parametros": melhores_modelos[melhor_nome]["parametros"],
        "normalizacao": "pixels divididos por 255.0",
    }
    caminho = MODELS_DIR / "melhor_modelo_mnist.joblib"
    joblib.dump(pacote, caminho)
    print(f"\nMelhor modelo salvo em: {caminho}")
    return caminho


def normalizar_nome(texto):
    """Cria nomes simples para arquivos."""
    return (
        str(texto)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def executar_pipeline():
    """Executa o projeto inteiro em sequência."""
    preparar_pastas()
    X, y = carregar_mnist()

    mostrar_distribuicao_classes(y, "distribuicao_classes_base_completa.png")
    mostrar_grade_digitos(X, y)

    if USAR_AMOSTRA_RAPIDA:
        print(
            f"\nUsando amostra rápida com até {AMOSTRA_POR_CLASSE} imagens por classe "
            "para facilitar os testes no VS Code."
        )
        X_modelo, y_modelo = criar_amostra_estratificada(X, y, AMOSTRA_POR_CLASSE)
    else:
        X_modelo, y_modelo = X, y

    mostrar_distribuicao_classes(y_modelo, "distribuicao_classes_modelagem.png")

    X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = dividir_e_normalizar(
        X_modelo, y_modelo
    )

    melhores_modelos, historico_df = treinar_modelos(
        X_treino,
        y_treino,
        X_validacao,
        y_validacao,
    )
    comparativo, predicoes = avaliar_modelos(melhores_modelos, X_teste, y_teste)

    melhor_nome = comparativo.iloc[0]["modelo"]
    digito_real, digito_previsto, total = identificar_maior_confusao(
        y_teste,
        predicoes[melhor_nome],
    )
    print(
        f"\nNo melhor modelo, a maior confusão foi: "
        f"dígito {digito_real} previsto como {digito_previsto}, com {total} casos."
    )

    resultado_ood = treinar_teste_ood(
        X_treino,
        y_treino,
        X_teste,
        y_teste,
        melhores_modelos,
        classes_ocultas=(4, 7),
    )

    salvar_melhor_modelo(melhores_modelos, comparativo)
    testar_imagens_proprias(melhores_modelos[melhor_nome]["modelo"])

    return {
        "historico": historico_df,
        "comparativo": comparativo,
        "melhores_modelos": melhores_modelos,
        "predicoes": predicoes,
        "resultado_ood": resultado_ood,
    }


if __name__ == "__main__":
    executar_pipeline()
