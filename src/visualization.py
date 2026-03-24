import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter

YIELD_COLS = ['1m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', '10y', '20y', '30y']

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

def plot_factor_heatmap(pcs, dates):
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
    writer = PillowWriter(fps=10)
    anim.save('yield_curve_animation.gif', writer=writer)
    print('Saved yield_curve_animation.gif')
    plt.close(fig)

def plot_short_rate_model_params(vasicek_df, cir_df):
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
