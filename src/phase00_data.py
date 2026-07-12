from pathlib import Path
import requests
import pandas as pd

URL = (
    "https://drive.google.com/file/d/1er0sPjY51RymKrRkzZkjFOVwJ8PEZoAZ/view?usp=sharing"
)

# Caminho da raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Caminho da pasta data/raw
RAW_DIR = ROOT_DIR / "data" / "raw"

# Arquivo de destino
DATASET_FILE = RAW_DIR / "manutencao_preditiva.csv"


def baixar_manutencao_preditiva(force: bool = False) -> Path:
    """
    Faz o download do dataset de manutenção preditiva.

    Parametros
    ----------
    force : bool, (opcional)
        Se True, substitui o arquivo existente.
        Se False, reutiliza o arquivo local caso ele já exista.

    Returns
    -------
    Path
        Caminho do arquivo CSV.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_FILE.exists() and not force:
        print(f"Utilizando dataset existente: {DATASET_FILE}")
        return DATASET_FILE

    print("Baixando dataset de manutenção preditiva...")

    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    DATASET_FILE.write_bytes(response.content)

    print(f"Dataset salvo em: {DATASET_FILE}")

    return DATASET_FILE

def carregar_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em:\n{DATASET_FILE}"
        )

    print(f"Carregando dataset: {DATASET_FILE.name}")

    df = pd.read_csv(DATASET_FILE)

    print(f"Dataset carregado com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    return df