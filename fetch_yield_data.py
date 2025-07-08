import requests
import pandas as pd
from io import StringIO

# FRED series IDs for Treasury yields (constant maturities)
FRED_SERIES = {
    '1m': 'DGS1MO',
    '3m': 'DGS3MO',
    '6m': 'DGS6MO',
    '1y': 'DGS1',
    '2y': 'DGS2',
    '3y': 'DGS3',
    '5y': 'DGS5',
    '7y': 'DGS7',
    '10y': 'DGS10',
    '20y': 'DGS20',
    '30y': 'DGS30',
}

FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'


def fetch_series(series_id):
    url = FRED_URL.format(series_id)
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ['DATE', series_id]
    return df


def fetch_all_yields():
    dfs = []
    for label, series_id in FRED_SERIES.items():
        df = fetch_series(series_id)
        df = df.rename(columns={series_id: label})
        dfs.append(df)
    # Merge on DATE
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='DATE', how='outer')
    merged = merged.sort_values('DATE').reset_index(drop=True)
    return merged


def main():
    print('Fetching US Treasury yield curve data from FRED...')
    df = fetch_all_yields()
    df.to_csv('yield_curve_data.csv', index=False)
    print('Saved yield_curve_data.csv with shape:', df.shape)


if __name__ == '__main__':
    main() 