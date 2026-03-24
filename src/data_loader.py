import pandas as pd
import numpy as np

YIELD_COLS = ['1m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', '10y', '20y', '30y']

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    df = df[YIELD_COLS]
    df = df.replace('.', np.nan).astype(float)
    df = df.dropna()
    return df
