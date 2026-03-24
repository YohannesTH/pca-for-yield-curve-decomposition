import pandas as pd
import matplotlib.pyplot as plt

def macro_linkage_analysis(pca_factors):
    macro = pd.read_csv('macro_data.csv', parse_dates=['DATE'])
    macro = macro.set_index('DATE')
    macro = macro.resample('ME').ffill()
    combined = pd.concat([pca_factors, macro], axis=1).ffill().dropna()
    print('\nCorrelation between PCA factors and macro variables:')
    print(combined.corr().loc[pca_factors.columns, macro.columns])
    
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
    event_dates = [
        '2008-09-15',
        '2020-03-15',
        '2022-03-16',
    ]
    window = 12
    for event in event_dates:
        event = pd.to_datetime(event)
        start = event - pd.DateOffset(months=window)
        end = event + pd.DateOffset(months=window)
        df = pd.concat([pca_factors, macro], axis=1).ffill().loc[start:end].dropna()
        if df.empty:
            continue
        print(f'\nEvent: {event.date()}')
        
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
        
        before = df.loc[start:event - pd.DateOffset(days=1)]
        after = df.loc[event + pd.DateOffset(days=1):end]
        print('Average factor change (after - before):')
        print((after.mean() - before.mean())[['PC1_Level', 'PC2_Slope', 'PC3_Curvature']])
