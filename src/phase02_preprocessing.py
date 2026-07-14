import pandas as pd


def identificar_duplicatas(df):

    df_duplicados = df[df.duplicated()]

    total_duplicados = len(df_duplicados)

    percentual_duplicados = (total_duplicados / len(df)) * 100

    print("=== Análise de Duplicatas ===")
    print(f"Total de linhas duplicadas: {total_duplicados}")
    print(f"Percentual de duplicatas: {percentual_duplicados:.2f}%")

    return df_duplicados

def remover_duplicatas(df):

    df_duplicados = identificar_duplicatas(df)

    total_duplicados = len(df_duplicados)

    if total_duplicados == 0:
        print("Nenhuma duplicata encontrada. Dataset mantido.")
        return df

    df_sem_duplicatas = df.drop_duplicates()

    print("=== Remoção de Duplicatas ===")
    print(f"Linhas removidas: {total_duplicados}")
    print(f"Dimensão anterior: {df.shape}")
    print(f"Dimensão posterior: {df_sem_duplicatas.shape}")

    return df_sem_duplicatas

def identificar_dados_ausentes(df):

    dados_ausentes = df.isnull().sum()

    percentual_ausentes = (dados_ausentes / len(df)) * 100

    resumo_ausentes = pd.DataFrame({
        "quantidade_ausentes": dados_ausentes,
        "percentual_ausentes": percentual_ausentes
    })

    # Mantém apenas colunas com valores ausentes
    resumo_ausentes = resumo_ausentes[
        resumo_ausentes["quantidade_ausentes"] > 0
    ]

    if resumo_ausentes.empty:
        print("Nenhum dado ausente encontrado.")
    else:
        print("=== Análise de Dados Ausentes ===")
        print(resumo_ausentes)

    return resumo_ausentes

def checar_distribuicao(df):
  
    # Seleciona apenas variáveis numéricas
    df_numerico = df.select_dtypes(include="number")

    distribuicao = pd.DataFrame({
        "media": df_numerico.mean(),
        "mediana": df_numerico.median(),
        "desvio_padrao": df_numerico.std(),
        "minimo": df_numerico.min(),
        "percentil_25": df_numerico.quantile(0.25),
        "percentil_50": df_numerico.quantile(0.50),
        "percentil_75": df_numerico.quantile(0.75),
        "maximo": df_numerico.max(),
        "assimetria": df_numerico.skew()
    })

    print("=== Distribuição das Variáveis Numéricas ===")
    display(distribuicao)

    return distribuicao

def definir_metodo_imputacao(distribuicao, dados_ausentes):
    """
    Define o método de imputação para cada variável com dados ausentes.

    Regras:
    - Percentual de dados ausentes < 10%
    - Assimetria entre -1 e 1 -> Média
    - Demais casos -> Mediana

    Retorna:
        DataFrame com o método de imputação de cada variável.
    """

    metodos = []

    for coluna in dados_ausentes.index:

        percentual = dados_ausentes.loc[coluna, "percentual_ausentes"]
        assimetria = distribuicao.loc[coluna, "assimetria"]

        if percentual >= 10:
            metodo = "Não recomendado (avaliar variável)"
        elif -1 <= assimetria <= 1:
            metodo = "Média"
        else:
            metodo = "Mediana"

        metodos.append({
            "variavel": coluna,
            "percentual_ausentes": percentual,
            "assimetria": assimetria,
            "metodo_imputacao": metodo
        })

    return pd.DataFrame(metodos)

def substituir_dados_ausentes(df, metodos_imputacao):

    # Cria uma cópia para preservar o DataFrame original (já sem os itens duplicados)
    df_processado = df.copy()

    print("=== Imputação de Dados Ausentes ===")

    for _, linha in metodos_imputacao.iterrows():

        coluna = linha["variavel"]
        metodo = linha["metodo_imputacao"]

        if metodo == "Média":
            valor_imputacao = df_processado[coluna].mean()

        elif metodo == "Mediana":
            valor_imputacao = df_processado[coluna].median()

        else:
            print(f"{coluna}: imputação não realizada ({metodo}).")
            continue

        quantidade = df_processado[coluna].isna().sum()

        df_processado[coluna] = df_processado[coluna].fillna(valor_imputacao)

        print(
            f"{coluna}: {quantidade} valor(es) substituído(s) "
            f"pela {metodo.lower()} ({valor_imputacao:.2f})"
        )

    print("\nImputação concluída.")

    return df_processado