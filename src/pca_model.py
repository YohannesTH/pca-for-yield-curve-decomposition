import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_pca(df, n_components=3):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(df)
    explained = pca.explained_variance_ratio_
    components = pca.components_
    return pcs, explained, components, pca

def save_factors_to_csv(pcs, dates):
    factor_df = pd.DataFrame(pcs[:, :3], index=dates, columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature'])
    factor_df.index.name = 'DATE'
    factor_df.to_csv('yield_curve_factors.csv')
    print('Saved PCA factor time series to yield_curve_factors.csv')

def rolling_pca_analysis(df, window=60):
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

def tsne_analysis(df):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(df.values)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=range(len(df)), cmap='viridis', s=10)
    plt.colorbar(scatter, label='Time (earlier to later)')
    plt.title('t-SNE Embedding of Yield Curves')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()

def statistical_analysis(pca_factors, ns_factors, explained):
    combined = pd.concat([pca_factors, ns_factors], axis=1).dropna()
    corr = combined.corr().loc[pca_factors.columns, ns_factors.columns]
    print('\nCorrelation between PCA and Nelson-Siegel factors:')
    print(corr)
    print('\nVariance explained by each PCA component:')
    for i, var in enumerate(explained):
        print(f'PC{i+1}: {var:.4f} ({var*100:.2f}%)')
