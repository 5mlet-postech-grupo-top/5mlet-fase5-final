import pandas as pd
from src.preprocessing import split_X_y


def test_split_X_y_builds_binary_target():
    df = pd.DataFrame({
        "IDADE_ALUNO_2020": [10, 11, 12],
        "ANOS_PM_2020": [1, 2, 3],
        "INDE_2020": [5.0, 6.0, 7.0],
        "FASE_TURMA_2020": ["1A", "1B", "2A"],
        "PEDRA_2020": ["Quartzo", "Ágata", "Ametista"],
        "INSTITUICAO_ENSINO_ALUNO_2020": ["X", "Y", "Z"],
        "DEFASAGEM_2021": [-1, 0, 2],
    })
    X, y = split_X_y(df)

    assert list(y) == [1, 0, 0]
    assert "DEFASAGEM_2021" not in X.columns
    assert "INDE_2020" in X.columns
