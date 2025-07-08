import requests
import pandas as pd
from io import StringIO

def fetch_fred_series(series_id):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ['DATE', series_id]
    return df

def main():
    # FRED series IDs
    series = {
        'GDP_GROWTH': 'A191RL1Q225SBEA',  # Real GDP: Percent Change from Preceding Period (quarterly)
        'CPI': 'CPIAUCSL',                # Consumer Price Index for All Urban Consumers: All Items (monthly)
        'FEDFUNDS': 'FEDFUNDS',           # Effective Federal Funds Rate (monthly)
    }
    dfs = []
    for label, sid in series.items():
        df = fetch_fred_series(sid)
        df = df.rename(columns={sid: label})
        dfs.append(df)
    # Merge on DATE
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='DATE', how='outer')
    merged = merged.sort_values('DATE').reset_index(drop=True)
    merged.to_csv('macro_data.csv', index=False)
    print('Saved macro_data.csv with shape:', merged.shape)

if __name__ == '__main__':
    main() 