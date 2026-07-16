import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

VARIAVEL_ALVO = "falha_maquina"

COLUNAS_DESCARTADAS = [
    "udi",
    "id_produto",
    "falha_twf",
    "falha_hdf",
    "falha_pwf",
    "falha_osf",
    "falha_rnf",
]

def separar_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
   
    x = df.drop(columns=COLUNAS_DESCARTADAS + [VARIAVEL_ALVO]) #X : DataFrame com as variáveis preditoras.

    y = df[VARIAVEL_ALVO] #y : Series com a variável alvo.

    print("=== Separação de Features e Target ===")
    print(f"Features (x): {x.shape}")
    print(f"Target   (y): {y.shape}")

    return x, y

from sklearn.model_selection import train_test_split

def separar_train_test(
    x,
    y,
    test_size=0.20,
    random_state=42
):
    
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print("=== Separação Treino/Teste ===")
    print(f"x_train: {x_train.shape}")
    print(f"x_test : {x_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    return x_train, x_test, y_train, y_test

def aplicar_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
):
    
    print("=== Balanceamento com SMOTE ===")

    print("\nDistribuição antes do SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=random_state)

    x_train_balanceado, y_train_balanceado = smote.fit_resample(
        x_train,
        y_train
    )

    print("\nDistribuição após o SMOTE:")
    print(y_train_balanceado.value_counts())

    print("\nBalanceamento concluído.")

    return x_train_balanceado, y_train_balanceado