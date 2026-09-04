import pandas as pd


def load_data(file_path):
    return pd.read_csv(
        file_path,
        na_values=["?"],
        keep_default_na=False
    )