import pandas as pd
import tempfile
from pathlib import Path

import pytest

from src.data_loader import list_xlsx, load_all_training_data


def test_list_xlsx_empty(tmp_path):
    # non-existent directory returns empty
    assert list_xlsx(tmp_path / "nonexistent") == []

    # directory with non-xlsx file
    p = tmp_path / "foo.txt"
    p.write_text("hi")
    assert list_xlsx(tmp_path) == []


def test_load_all_training_data_success(tmp_path, monkeypatch):
    # create two xlsx files
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3]})
    f1 = tmp_path / "one.xlsx"
    f2 = tmp_path / "two.xlsx"
    df1.to_excel(f1, index=False)
    df2.to_excel(f2, index=False)

    # use tmp_path as data dir
    result = load_all_training_data(tmp_path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result.columns) >= {"a", "__source_file__"}


def test_load_all_training_data_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_training_data(tmp_path)
