import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def gerar_predicoes_knn(
    n_neighbors,
    x_train,
    y_train,
    x_test
):

    modelo = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    modelo.fit(
        x_train,
        y_train
    )

    return modelo.predict(x_test)

def gerar_predicoes_arvore(
    max_depth,
    x_train,
    y_train,
    x_test
):

    modelo = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    modelo.fit(
        x_train,
        y_train
    )

    return modelo.predict(x_test)

def calcular_acuracia(
    y_real,
    y_pred
):

    acuracia = accuracy_score(
        y_real,
        y_pred
    )

    return acuracia

def criar_tabela_comparacao(
    acuracia_knn,
    acuracia_arvore
):

    tabela = pd.DataFrame({
        "Modelo": [
            "KNN",
            "Árvore de Decisão"
        ],
        "Acurácia": [
            acuracia_knn,
            acuracia_arvore
        ]
    })

    return tabela.sort_values(
        by="Acurácia",
        ascending=False
    ).reset_index(drop=True)