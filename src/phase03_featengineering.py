import pandas as pd
import matplotlib.pyplot as plt

def criar_feature_potencia(df):
   

    df_processado = df.copy()

    df_processado["potencia"] = (
        df_processado["velocidade_rotacao_rpm"] *
        df_processado["torque_nm"]
    )

    return df_processado

def criar_delta_temp_k(df):
   

    df = df.copy()

    df["delta_temp"] = (
        df["temperatura_processo_k"] -
        df["temperatura_ar_k"]
    )

    return df