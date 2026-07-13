import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def mostrar_dimensoes(df):
    
    linhas, colunas = df.shape

    print("=== Dimensões do Dataset ===")
    print(f"Linhas : {linhas}")
    print(f"Colunas: {colunas}")

def mostrar_tipos(df):

    print("=== Tipos das Variáveis ===")
    print(df.dtypes)

def mostrar_estatisticas(df):

    print("=== Estatísticas Descritivas ===")
    return df.describe()

def mostrar_graficos(df):

    PREDITORAS = [
        "temperatura_ar_k",
        "temperatura_processo_k",
        "velocidade_rotacao_rpm",
        "torque_nm",
        "desgaste_ferramenta_min",
    ]

    scatter_matrix(
        df[PREDITORAS],
        figsize=(12, 12),
        diagonal="hist",
        alpha=0.4
    )

    plt.show()

def plotar_balanceamento(df):
    
    contagem = df["falha_maquina"].value_counts().sort_index()

    plt.figure(figsize=(6,4))

    barras = plt.bar(
        ["Sem Falha", "Com Falha"],
        contagem
    )

    total = contagem.sum()

    for barra in barras:
        altura = barra.get_height()
        percentual = altura / total * 100

        plt.text(
            barra.get_x() + barra.get_width()/2,
            altura,
            f"{percentual:.1f}%",
            ha="center",
            va="bottom"
        )

    plt.title("Balanceamento da Variável Alvo")
    plt.xlabel("Estado da Máquina")
    plt.ylabel("Quantidade de Registros")

    plt.show()



def plotar_correlacao(df):

    PREDITORAS = [
        "temperatura_ar_k",
        "temperatura_processo_k",
        "velocidade_rotacao_rpm",
        "torque_nm",
        "desgaste_ferramenta_min",
        "falha_maquina"
    ]

    correlacao = df[PREDITORAS].corr(method="pearson")

    plt.figure(figsize=(8,6))

    sns.heatmap(
        correlacao,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Mapa de Calor - Correlação de Pearson")

    plt.show()