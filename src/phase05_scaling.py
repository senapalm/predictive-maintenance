import pandas as pd
from sklearn.preprocessing import StandardScaler


def identificar_variaveis_scaling(df, variavel_alvo, colunas_excluir=None):

    # Seleciona todas as colunas numéricas
    variaveis_scaling = df.select_dtypes(include="number").columns.tolist()

    # Remove a variável alvo
    variaveis_scaling = [
        coluna
        for coluna in variaveis_scaling
        if coluna != variavel_alvo
    ]

    # Remove variáveis binárias (dummy)
    variaveis_scaling = [
        coluna
        for coluna in variaveis_scaling
        if df[coluna].nunique() > 2
    ]

    # Remove variáveis numéricas que não são contínuas
    if colunas_excluir is None:
        colunas_excluir = []

    variaveis_scaling = [
        coluna
        for coluna in variaveis_scaling
        if coluna not in colunas_excluir
    ]

    return variaveis_scaling

def criar_scaler():

    scaler = StandardScaler()

    return scaler

def aplicar_scaling_treino(x_train, variaveis_scaling, scaler):


    x_train = x_train.copy() #protege o df de ser alterado

    x_train[variaveis_scaling] = scaler.fit_transform(
        x_train[variaveis_scaling]
    )

    return x_train

def aplicar_scaling_teste(x_test, variaveis_scaling, scaler):

    x_test = x_test.copy() #protege os dados para não alterar

    x_test[variaveis_scaling] = scaler.transform(
        x_test[variaveis_scaling]
    )

    return x_test

def validar_scaling(x_train, variaveis_scaling):

    validacao = x_train[variaveis_scaling].agg(["mean", "std"])

    return validacao