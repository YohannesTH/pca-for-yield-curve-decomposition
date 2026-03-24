import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

def forecast_and_reconstruct(df, pcs, components, pca):
    factors = pd.DataFrame(pcs[:, :3], index=df.index, columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature'])
    model = VAR(factors)
    results = model.fit(1)
    print('\nVAR(1) summary:')
    print(results.summary())
    lag_order = results.k_ar
    forecast = results.forecast(factors.values[-lag_order:], steps=12)
    forecast_idx = pd.date_range(factors.index[-1] + pd.DateOffset(months=1), periods=12, freq='ME')
    forecast_df = pd.DataFrame(forecast, index=forecast_idx, columns=factors.columns)
    forecast_df.to_csv('forecasted_factors.csv')
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    def reconstruct_yield(factor_row):
        return np.dot(factor_row.values, components[:3]) + pca.mean_
    actual_yield = reconstruct_yield(factors.iloc[-1])
    forecasted_yields = np.vstack([reconstruct_yield(forecast_df.iloc[i]) for i in range(12)])
    forecasted_yield_df = pd.DataFrame(forecasted_yields, index=forecast_idx, columns=[str(m) for m in maturities])
    forecasted_yield_df.to_csv('forecasted_yield_curves.csv')
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, actual_yield, marker='o', color='k', label=f'Actual ({factors.index[-1].date()})')
    for i in range(12):
        plt.plot(maturities, forecasted_yields[i], marker='.', alpha=0.6, label=f'Forecast {forecast_idx[i].strftime("%Y-%m")}' if i in [0, 5, 11] else None)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.title('Actual and Forecasted Yield Curves (Next 12 Months)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
