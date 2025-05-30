import pandas as pd
import pytest

from features.preprocessing import preprocess_data


def test_preprocess_data_valid():
    df = pd.DataFrame(
        {
            "TotalCharges": ["100.0", "200.0", "300.0"],
            "Churn": ["Yes", "No", "Yes"],
            "Category": ["A", "B", "A"],
        }
    )
    result = preprocess_data(df)
    assert "TotalCharges" in result.columns
    assert "Churn" in result.columns
    assert any(col.startswith("Category_") for col in result.columns)
    assert not result.isnull().values.any()


def test_preprocess_data_missing_columns():
    df = pd.DataFrame({"TotalCharges": ["10", "20"]})
    with pytest.raises(KeyError):
        preprocess_data(df)


def test_preprocess_data_blank_totalcharges():
    df = pd.DataFrame({"TotalCharges": [" ", " "], "Churn": ["Yes", "No"]})
    with pytest.raises(ValueError):
        preprocess_data(df)


def test_preprocess_data_handles_none_category():
    # Kontrollera att None i en kategori inte ger NaN
    df = pd.DataFrame(
        {
            "TotalCharges": ["100", "200"],
            "Churn": ["Yes", "No"],
            "Payment": [None, "Credit"],
        }
    )
    result = preprocess_data(df)
    assert not result.isnull().values.any()
