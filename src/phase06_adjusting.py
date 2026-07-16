import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def avaliar_knn_parametros(
    x_train,
    x_test,
    y_train,
    y_test,
    lista_k=[3, 5, 7]
):
    print("=== Avaliação dos Parâmetros do KNN ===")

    resultados = []

    for k in lista_k:

        modelo = KNeighborsClassifier(n_neighbors=k)

        modelo.fit(x_train, y_train)

        acuracia_treino = modelo.score(x_train, y_train)
        acuracia_teste = modelo.score(x_test, y_test)

        resultados.append({
            "k": k,
            "acuracia_treino": acuracia_treino,
            "acuracia_teste": acuracia_teste
        })

    resultado = pd.DataFrame(resultados)

    resultado = resultado.sort_values("k")

    resultado[["acuracia_treino", "acuracia_teste"]] = (
        resultado[["acuracia_treino", "acuracia_teste"]]
        .round(4)
    )

    return resultado

from sklearn.tree import DecisionTreeClassifier
import pandas as pd


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def avaliar_arvore_parametros(
    x_train,
    x_test,
    y_train,
    y_test,
    lista_max_depth=[3, 5, None]
):
    print("=== Avaliação dos Parâmetros da Árvore de Decisão ===")

    resultados = []

    for max_depth in lista_max_depth:

        modelo = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )

        modelo.fit(x_train, y_train)

        acuracia_treino = modelo.score(x_train, y_train)
        acuracia_teste = modelo.score(x_test, y_test)

        resultados.append({
            "max_depth": "None" if max_depth is None else max_depth,
            "acuracia_treino": acuracia_treino,
            "acuracia_teste": acuracia_teste
        })

    resultado = pd.DataFrame(resultados)

    resultado[["acuracia_treino", "acuracia_teste"]] = (
        resultado[["acuracia_treino", "acuracia_teste"]]
        .round(4)
    )

    return resultado

def calcular_diferenca_overfitting(resultado, nome_modelo="resultado"):

    print(f"=== Cálculo da Diferença de Overfitting - {nome_modelo} ===")

    resultado = resultado.copy()

    resultado["diferenca_overfitting"] = (
        resultado["acuracia_treino"] -
        resultado["acuracia_teste"]
    ).round(4)

    return resultado
