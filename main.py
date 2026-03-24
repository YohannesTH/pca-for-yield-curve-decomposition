import pandas as pd

from src.data_loader import load_and_clean_data
from src.pca_model import (run_pca, save_factors_to_csv,
                           rolling_pca_analysis, tsne_analysis,
                           statistical_analysis)
from src.parametric_models import (fit_nelson_siegel_all, 
                                   fit_svensson_all, 
                                   fit_vasicek_cir_all)
from src.macro_analysis import macro_linkage_analysis, event_study
from src.forecasting import forecast_and_reconstruct
from src.visualization import (plot_pca_components, plot_factor_time_series,
                               plot_comparison, plot_model_comparison,
                               plot_factor_heatmap, animate_yield_curve,
                               plot_short_rate_model_params)

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
    
    print('Fitting Nelson-Siegel model to each date...')
    ns_factors = fit_nelson_siegel_all(df)
    pca_factors = pd.read_csv('yield_curve_factors.csv', index_col='DATE', parse_dates=True)
    plot_comparison(pca_factors, ns_factors)
    
    statistical_analysis(pca_factors, ns_factors, explained)
    
    macro_linkage_analysis(pca_factors)
    
    macro = pd.read_csv('macro_data.csv', parse_dates=['DATE']).set_index('DATE').resample('ME').ffill()
    event_study(pca_factors, macro)
    
    forecast_and_reconstruct(df, pcs, components, pca)
    
    plot_factor_heatmap(pcs, df.index)
    animate_yield_curve(df)
    rolling_pca_analysis(df)
    
    print('Fitting Svensson model to each date...')
    sv_factors = fit_svensson_all(df)
    plot_model_comparison(pca_factors, ns_factors, sv_factors)
    
    print('Fitting Vasicek and CIR models to each date...')
    vasicek_df, cir_df = fit_vasicek_cir_all(df)
    plot_short_rate_model_params(vasicek_df, cir_df)
    
    tsne_analysis(df)

if __name__ == '__main__':
    main()
