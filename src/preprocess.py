# src/preprocess.py
import pandas as pd

LABEL_MAP = {"sunny": 0, "cloudy": 1, "rain": 2}

def add_label_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_id"] = df["label"].map(LABEL_MAP).astype("int8")
    return df
