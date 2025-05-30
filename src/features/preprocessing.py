import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rensar och kodar churn-dataset för modellering.

    Gör följande:
      - Verifierar att nödvändiga kolumner finns
      - Tar bort rader där 'TotalCharges' är tom sträng
      - Konverterar 'TotalCharges' till float
      - Mappar 'Churn' från Yes/No till 1/0
      - One-hot-encodar alla kvarvarande kategoriska kolumner (drop_first=True)
      - Kontrollerar att inga null-värden finns kvar

    Args:
        df: pandas DataFrame som innehåller minst kolumnerna
            ['TotalCharges', 'Churn']

    Returns:
        En ny DataFrame som är färdig för träning eller prediktion.

    Raises:
        KeyError: Om någon av kolumnerna 'TotalCharges' eller 'Churn' saknas.
        ValueError: Om DataFrame är tom efter rensning eller innehåller
        null-värden.
    """
    # Kontrollera obligatoriska kolumner
    required = ["TotalCharges", "Churn"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

    # Rensa bort rader med tomma TotalCharges
    df_clean = df[df["TotalCharges"] != " "].copy()
    if df_clean.empty:
        raise ValueError("No rows left after removing blank TotalCharges")

    # Konvertera typer
    df_clean["TotalCharges"] = df_clean["TotalCharges"].astype(float)
    df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    # One-hot-encoda kategorier (drop_first för att undvika dummy-fälla)
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    # Kontrollera att inga null-värden återstår
    if df_encoded.isnull().values.any():
        raise ValueError("Null values detected after preprocessing")

    return df_encoded
