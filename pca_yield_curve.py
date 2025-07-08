import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from statsmodels.tsa.api import VAR
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.manifold import TSNE

YIELD_COLS = ['1m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', '10y', '20y', '30y']


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    df = df[YIELD_COLS]
    df = df.replace('.', np.nan).astype(float)
    df = df.dropna()
    return df


def run_pca(df, n_components=3):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(df)
    explained = pca.explained_variance_ratio_
    components = pca.components_
    return pcs, explained, components, pca


def plot_pca_components(components):
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(['Level', 'Slope', 'Curvature']):
        plt.plot(YIELD_COLS, components[i], marker='o', label=f'PC{i+1} ({label})')
    plt.title('First 3 Principal Components of Yield Curve')
    plt.xlabel('Maturity')
    plt.ylabel('Loading')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_factor_time_series(pcs, dates):
    plt.figure(figsize=(12, 8))
    labels = ['Level', 'Slope', 'Curvature']
    for i in range(3):
        plt.plot(dates, pcs[:, i], label=labels[i])
    plt.title('Time Series of First 3 Principal Component Scores (Yield Curve Factors)')
    plt.xlabel('Date')
    plt.ylabel('Factor Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_factors_to_csv(pcs, dates):
    factor_df = pd.DataFrame(pcs[:, :3], index=dates, columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature'])
    factor_df.index.name = 'DATE'
    factor_df.to_csv('yield_curve_factors.csv')
    print('Saved PCA factor time series to yield_curve_factors.csv')


# Nelson-Siegel model function
def nelson_siegel(maturities, beta0, beta1, beta2, tau):
    maturities = np.array(maturities)
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-maturities / tau)) / (maturities / tau)
    term3 = beta2 * ((1 - np.exp(-maturities / tau)) / (maturities / tau) - np.exp(-maturities / tau))
    return term1 + term2 + term3


# Fit Nelson-Siegel to each date's yield curve
def fit_nelson_siegel_all(df):
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  # in years
    ns_params = []
    for row in df.values:
        try:
            popt, _ = curve_fit(
                nelson_siegel, maturities, row,
                p0=[row.mean(), -1, 1, 2],
                bounds=([-np.inf, -np.inf, -np.inf, 0.05], [np.inf, np.inf, np.inf, 10])
            )
        except Exception:
            popt = [np.nan, np.nan, np.nan, np.nan]
        ns_params.append(popt)
    ns_params = np.array(ns_params)
    ns_df = pd.DataFrame(ns_params[:, :3], index=df.index, columns=['NS_Level', 'NS_Slope', 'NS_Curvature'])
    return ns_df


def svensson(maturities, beta0, beta1, beta2, beta3, tau1, tau2):
    maturities = np.array(maturities)
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-maturities / tau1)) / (maturities / tau1)
    term3 = beta2 * ((1 - np.exp(-maturities / tau1)) / (maturities / tau1) - np.exp(-maturities / tau1))
    term4 = beta3 * ((1 - np.exp(-maturities / tau2)) / (maturities / tau2) - np.exp(-maturities / tau2))
    return term1 + term2 + term3 + term4


def fit_svensson_all(df):
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    sv_params = []
    for row in df.values:
        try:
            popt, _ = curve_fit(
                svensson, maturities, row,
                p0=[row.mean(), -1, 1, 1, 2, 0.5],
                bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0.05, 0.05], [np.inf, np.inf, np.inf, np.inf, 10, 10])
            )
        except Exception:
            popt = [np.nan]*6
        sv_params.append(popt)
    sv_params = np.array(sv_params)
    sv_df = pd.DataFrame(sv_params[:, :4], index=df.index, columns=['SV_Level', 'SV_Slope', 'SV_Curvature', 'SV_Extra'])
    return sv_df


def plot_model_comparison(pca_factors, ns_factors, sv_factors):
    labels = ['Level', 'Slope', 'Curvature']
    plt.figure(figsize=(14, 12))
    for i, label in enumerate(labels):
        plt.subplot(3, 1, i+1)
        plt.plot(pca_factors.index, pca_factors.iloc[:, i], label='PCA ' + label)
        plt.plot(ns_factors.index, ns_factors.iloc[:, i], label='Nelson-Siegel ' + label, alpha=0.7)
        plt.plot(sv_factors.index, sv_factors.iloc[:, i], label='Svensson ' + label, alpha=0.7)
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
    plt.xlabel('Date')
    plt.suptitle('Comparison of PCA, Nelson-Siegel, and Svensson Yield Curve Factors')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def plot_comparison(pca_factors, ns_factors):
    plt.figure(figsize=(14, 10))
    labels = ['Level', 'Slope', 'Curvature']
    for i, label in enumerate(labels):
        plt.subplot(3, 1, i+1)
        plt.plot(pca_factors.index, pca_factors.iloc[:, i], label='PCA ' + label)
        plt.plot(ns_factors.index, ns_factors.iloc[:, i], label='Nelson-Siegel ' + label, alpha=0.7)
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
    plt.xlabel('Date')
    plt.suptitle('Comparison of PCA and Nelson-Siegel Yield Curve Factors')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def statistical_analysis(pca_factors, ns_factors, explained):
    # Align indices
    combined = pd.concat([pca_factors, ns_factors], axis=1).dropna()
    corr = combined.corr().loc[pca_factors.columns, ns_factors.columns]
    print('\nCorrelation between PCA and Nelson-Siegel factors:')
    print(corr)
    # Variance explained
    print('\nVariance explained by each PCA component:')
    for i, var in enumerate(explained):
        print(f'PC{i+1}: {var:.4f} ({var*100:.2f}%)')


def macro_linkage_analysis(pca_factors):
    # Load macro data
    macro = pd.read_csv('macro_data.csv', parse_dates=['DATE'])
    macro = macro.set_index('DATE')
    # Resample macro data to monthly (GDP is quarterly, forward fill)
    macro = macro.resample('M').ffill()
    # Align with PCA factors
    combined = pd.concat([pca_factors, macro], axis=1).dropna()
    print('\nCorrelation between PCA factors and macro variables:')
    print(combined.corr().loc[pca_factors.columns, macro.columns])
    # Plot time series
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(['PC1_Level', 'PC2_Slope', 'PC3_Curvature']):
        plt.subplot(3, 1, i+1)
        plt.plot(combined.index, combined[label], label=label)
        plt.ylabel(label)
        plt.legend(loc='upper left')
        plt.grid(True)
        if i == 0:
            plt.twinx()
            plt.plot(combined.index, combined['GDP_GROWTH'], color='tab:orange', label='GDP_GROWTH', alpha=0.5)
            plt.ylabel('GDP_GROWTH')
        elif i == 1:
            plt.twinx()
            plt.plot(combined.index, combined['CPI'], color='tab:green', label='CPI', alpha=0.5)
            plt.ylabel('CPI')
        elif i == 2:
            plt.twinx()
            plt.plot(combined.index, combined['FEDFUNDS'], color='tab:red', label='FEDFUNDS', alpha=0.5)
            plt.ylabel('FEDFUNDS')
    plt.suptitle('PCA Factors and Macroeconomic Variables')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def event_study(pca_factors, macro):
    import matplotlib.dates as mdates
    # Example event dates (Fed rate hikes, financial crisis, etc.)
    event_dates = [
        '2008-09-15',  # Lehman collapse
        '2020-03-15',  # COVID Fed emergency cut
        '2022-03-16',  # Start of 2022 hiking cycle
    ]
    window = 12  # months before/after
    for event in event_dates:
        event = pd.to_datetime(event)
        start = event - pd.DateOffset(months=window)
        end = event + pd.DateOffset(months=window)
        df = pd.concat([pca_factors, macro], axis=1).loc[start:end]
        if df.empty:
            continue
        print(f'\nEvent: {event.date()}')
        # Plot
        plt.figure(figsize=(12, 7))
        for i, label in enumerate(['PC1_Level', 'PC2_Slope', 'PC3_Curvature']):
            plt.plot(df.index, df[label], label=label)
        plt.axvline(event, color='k', linestyle='--', label='Event')
        plt.title(f'PCA Factors around {event.date()}')
        plt.xlabel('Date')
        plt.ylabel('Factor Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # Print average before/after
        before = df.loc[start:event - pd.DateOffset(days=1)]
        after = df.loc[event + pd.DateOffset(days=1):end]
        print('Average factor change (after - before):')
        print((after.mean() - before.mean())[['PC1_Level', 'PC2_Slope', 'PC3_Curvature']])


def forecast_and_reconstruct(df, pcs, components, pca):
    import matplotlib.pyplot as plt
    # Prepare factor DataFrame
    factors = pd.DataFrame(pcs[:, :3], index=df.index, columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature'])
    # Fit VAR(1) model
    model = VAR(factors)
    results = model.fit(1)
    print('\nVAR(1) summary:')
    print(results.summary())
    # Forecast next 12 months
    lag_order = results.k_ar
    forecast = results.forecast(factors.values[-lag_order:], steps=12)
    forecast_idx = pd.date_range(factors.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    forecast_df = pd.DataFrame(forecast, index=forecast_idx, columns=factors.columns)
    # Save forecasted factors
    forecast_df.to_csv('forecasted_factors.csv')
    # Reconstruct yield curves for all forecasted months
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    def reconstruct_yield(factor_row):
        return np.dot(factor_row.values, components[:3]) + pca.mean_
    actual_yield = reconstruct_yield(factors.iloc[-1])
    forecasted_yields = np.vstack([reconstruct_yield(forecast_df.iloc[i]) for i in range(12)])
    # Save forecasted yield curves
    forecasted_yield_df = pd.DataFrame(forecasted_yields, index=forecast_idx, columns=[str(m) for m in maturities])
    forecasted_yield_df.to_csv('forecasted_yield_curves.csv')
    # Plot all forecasted yield curves
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


def plot_factor_heatmap(pcs, dates):
    import seaborn as sns
    import matplotlib.pyplot as plt
    factor_df = pd.DataFrame(pcs[:, :3], index=dates, columns=['Level', 'Slope', 'Curvature'])
    plt.figure(figsize=(14, 6))
    sns.heatmap(factor_df.T, cmap='coolwarm', center=0, cbar_kws={'label': 'Factor Value'})
    plt.title('Heatmap of PCA Factor Scores Over Time')
    plt.xlabel('Date')
    plt.ylabel('Factor')
    plt.tight_layout()
    plt.show()


def animate_yield_curve(df):
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], 'o-', lw=2)
    ax.set_xlim(maturities[0], maturities[-1])
    ax.set_ylim(df.min().min() - 0.5, df.max().max() + 0.5)
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Yield (%)')
    ax.set_title('Yield Curve Evolution (Animated)')
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(frame):
        y = df.iloc[frame].values
        line.set_data(maturities, y)
        ax.set_title(f'Yield Curve: {df.index[frame].strftime("%Y-%m")}')
        return line,
    
    anim = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=60)
    # Save animation
    writer = FFMpegWriter(fps=10)
    anim.save('yield_curve_animation.mp4', writer=writer)
    print('Saved yield_curve_animation.mp4')
    plt.close(fig)


def rolling_pca_analysis(df, window=60):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    maturities = ['1m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', '10y', '20y', '30y']
    loadings = []
    dates = []
    for i in range(window, len(df)):
        subdf = df.iloc[i-window:i]
        pca = PCA(n_components=1)
        pca.fit(subdf)
        loadings.append(pca.components_[0])
        dates.append(df.index[i])
    loadings = np.array(loadings)
    plt.figure(figsize=(12, 6))
    for j, mat in enumerate(maturities):
        plt.plot(dates, loadings[:, j], label=mat)
    plt.title(f'Rolling PCA (window={window} months): First PC Loadings Over Time')
    plt.xlabel('Date')
    plt.ylabel('PC1 Loading')
    plt.legend(title='Maturity')
    plt.tight_layout()
    plt.show()


def fit_vasicek_cir_all(df):
    from scipy.optimize import curve_fit
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    vasicek_params = []
    cir_params = []
    def vasicek_yield(m, kappa, theta, sigma, r0):
        B = (1 - np.exp(-kappa * m)) / kappa
        A = np.exp((theta - sigma**2/(2*kappa**2)) * (B - m) - (sigma**2 * B**2) / (4 * kappa))
        return -np.log(A) / m + B * r0 / m
    def cir_yield(m, kappa, theta, sigma, r0):
        gamma = np.sqrt(kappa**2 + 2 * sigma**2)
        exp_gamma_m = np.exp(gamma * m)
        numerator = 2 * gamma * np.exp((kappa + gamma) * m / 2)
        denominator = (gamma + kappa) * (exp_gamma_m - 1) + 2 * gamma
        B = 2 * (exp_gamma_m - 1) / denominator
        A = (numerator / denominator) ** (2 * kappa * theta / sigma**2)
        return -np.log(A) / m + B * r0 / m
    for row in df.values:
        # Vasicek
        try:
            popt, _ = curve_fit(lambda m, kappa, theta, sigma, r0: vasicek_yield(m, kappa, theta, sigma, r0),
                                maturities, row, p0=[0.1, 0.03, 0.01, row[0]],
                                bounds=([0.001, 0, 0, -1], [2, 0.2, 0.2, 1]))
        except Exception:
            popt = [np.nan]*4
        vasicek_params.append(popt)
        # CIR
        try:
            popt, _ = curve_fit(lambda m, kappa, theta, sigma, r0: cir_yield(m, kappa, theta, sigma, r0),
                                maturities, row, p0=[0.1, 0.03, 0.01, row[0]],
                                bounds=([0.001, 0, 0, -1], [2, 0.2, 0.2, 1]))
        except Exception:
            popt = [np.nan]*4
        cir_params.append(popt)
    vasicek_params = np.array(vasicek_params)
    cir_params = np.array(cir_params)
    vasicek_df = pd.DataFrame(vasicek_params[:, :2], index=df.index, columns=['Vasicek_kappa', 'Vasicek_theta'])
    cir_df = pd.DataFrame(cir_params[:, :2], index=df.index, columns=['CIR_kappa', 'CIR_theta'])
    return vasicek_df, cir_df


def plot_short_rate_model_params(vasicek_df, cir_df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(vasicek_df.index, vasicek_df['Vasicek_kappa'], label='Vasicek kappa')
    plt.plot(cir_df.index, cir_df['CIR_kappa'], label='CIR kappa', alpha=0.7)
    plt.ylabel('Mean Reversion Speed (kappa)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(vasicek_df.index, vasicek_df['Vasicek_theta'], label='Vasicek theta')
    plt.plot(cir_df.index, cir_df['CIR_theta'], label='CIR theta', alpha=0.7)
    plt.ylabel('Long-term Mean (theta)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.suptitle('Short-Rate Model Parameters Over Time (Vasicek & CIR)')
    plt.show()


def tsne_analysis(df):
    # Apply t-SNE to yield curve data
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(df.values)
    # Plot t-SNE embedding colored by time
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=range(len(df)), cmap='viridis', s=10)
    plt.colorbar(scatter, label='Time (earlier to later)')
    plt.title('t-SNE Embedding of Yield Curves')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()


def main():
    df = load_and_clean_data('yield_curve_data.csv')
    print(f'Loaded data shape after cleaning: {df.shape}')
    pcs, explained, components, pca = run_pca(df)
    print('Explained variance ratios (first 3 PCs):', explained)
    print('\nInterpretation:')
    print('PC1 (Level): Parallel shift of the yield curve (overall interest rate level).')
    print('PC2 (Slope): Difference between short and long rates (steepness of the curve).')
    print('PC3 (Curvature): Bending of the curve (mid-term rates vs. short/long).')
    plot_pca_components(components)
    plot_factor_time_series(pcs, df.index)
    save_factors_to_csv(pcs, df.index)
    # Fit Nelson-Siegel and compare
    print('Fitting Nelson-Siegel model to each date...')
    ns_factors = fit_nelson_siegel_all(df)
    pca_factors = pd.read_csv('yield_curve_factors.csv', index_col='DATE', parse_dates=True)
    plot_comparison(pca_factors, ns_factors)
    # Statistical analysis
    statistical_analysis(pca_factors, ns_factors, explained)
    # Macroeconomic linkage analysis
    macro_linkage_analysis(pca_factors)
    # Event study
    macro = pd.read_csv('macro_data.csv', parse_dates=['DATE']).set_index('DATE').resample('M').ffill()
    event_study(pca_factors, macro)
    # Forecasting and yield curve reconstruction
    forecast_and_reconstruct(df, pcs, components, pca)
    # Visualization enhancement: heatmap of factor scores
    plot_factor_heatmap(pcs, df.index)
    # Animated yield curve visualization
    animate_yield_curve(df)
    # Robustness check: rolling PCA
    rolling_pca_analysis(df)
    # Model comparison: Svensson
    print('Fitting Svensson model to each date...')
    sv_factors = fit_svensson_all(df)
    plot_model_comparison(pca_factors, ns_factors, sv_factors)
    # Model comparison: Vasicek and CIR
    print('Fitting Vasicek and CIR models to each date...')
    vasicek_df, cir_df = fit_vasicek_cir_all(df)
    plot_short_rate_model_params(vasicek_df, cir_df)
    # Non-linear dimensionality reduction: t-SNE
    tsne_analysis(df)


if __name__ == '__main__':
    main() 