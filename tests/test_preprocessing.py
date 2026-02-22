import pandas as pd
import numpy as np
from src.preprocessing import split_X_y
from src.preprocessing import enforce_types


def test_split_X_y_builds_binary_target():
    df = pd.DataFrame({
        "IDADE_ALUNO_2020": [10, 11, 12],
        "ANOS_PM_2020": [1, 2, 3],
        "INDE_2020": [5.0, 6.0, 7.0],
        "IEG_2020": [8.0, 9.0, 10.0],
        "FASE_TURMA_2020": ["1A", "1B", "2A"],
        "PEDRA_2020": ["Quartzo", "Ágata", "Ametista"],
        "INSTITUICAO_ENSINO_ALUNO_2020": ["X", "Y", "Z"],
        "DEFASAGEM_2021": [-1, 0, 2],
    })
    X, y = split_X_y(df)

    assert list(y) == [1, 0, 0]
    assert "DEFASAGEM_2021" not in X.columns
    assert "INDE" in X.columns
    assert "INDE_2020" not in X.columns


def test_enforce_types_limpeza_de_virgulas():
    """Garante que as notas com vírgula no padrão brasileiro são convertidas para float."""
    # Cria um DataFrame simulando o input sujo do utilizador
    df_sujo = pd.DataFrame({
        "INDE": ["8,5", "9,1", "7,0"],
        "FASE_TURMA": ["5G", "6A", "7B"]
    })

    df_limpo = enforce_types(df_sujo)

    # Verifica se a coluna INDE passou a ser numérica (float) e calculou corretamente
    assert pd.api.types.is_numeric_dtype(df_limpo["INDE"])
    assert df_limpo["INDE"].iloc[0] == 8.5


def test_enforce_types_tratamento_de_lixo():
    """Garante que valores absurdos em colunas numéricas viram NaN (Nulos)."""
    df_sujo = pd.DataFrame({
        "IDA": ["10", "ErroDeDigitacao", "6825-05-01"],
        "PEDRA": ["Ametista", "Topázio", "Quartzo"]
    })

    df_limpo = enforce_types(df_sujo)

    # Verifica se o texto virou NaN na coluna numérica
    assert pd.isna(df_limpo["IDA"].iloc[1])
    assert pd.isna(df_limpo["IDA"].iloc[2])
    # Verifica se a coluna categórica permaneceu intacta
    assert df_limpo["PEDRA"].iloc[0] == "Ametista"
