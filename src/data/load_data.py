from pathlib import Path

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Läser in en CSV-datafil och validerar att den är korrekt.

    Args:
        path: Sökväg till CSV-filen, relativt projektroten.
    Returns:
        En pandas DataFrame med innehållet i filen.
    Raises:
        FileNotFoundError: Om filen inte finns.
        RuntimeError: Om inläsningen av CSV misslyckas.
        ValueError: Om DataFrame är tom efter inläsning.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path.resolve()}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")

    if df.empty:
        raise ValueError(f"Loaded data is empty: {file_path.name}")

    return df
