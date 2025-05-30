import pandas as pd
import pytest

from data.load_data import load_data


def test_load_data_valid(tmp_path):
    # Skapa en temporär CSV-fil
    file = tmp_path / "mock.csv"
    file.write_text("col1,col2\n1,2\n3,4")
    df = load_data(str(file))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("does_not_exist.csv")


def test_load_data_empty(tmp_path):
    # Skapa fil som bara har header
    file = tmp_path / "empty.csv"
    file.write_text("col1,col2\n")
    with pytest.raises(ValueError):
        load_data(str(file))


def test_load_data_parse_error(tmp_path, monkeypatch):
    # Simulera fel vid pd.read_csv
    def bad_read(_):
        raise Exception("parse error")

    monkeypatch.setattr(pd, "read_csv", bad_read)
    file = tmp_path / "any.csv"
    file.write_text("a,b\n1,2")
    with pytest.raises(RuntimeError):
        load_data(str(file))
