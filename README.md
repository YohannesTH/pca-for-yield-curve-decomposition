# PCA for Yield Curve Decomposition

This project applies **Principal Component Analysis (PCA)** to decompose the **US Treasury yield curve** into its main components — **level**, **slope**, and **curvature** — and compares these with parametric models (Nelson-Siegel, Svensson, Vasicek, CIR). It also includes macroeconomic linkage analysis, VAR-based forecasting, and animated visualizations.

## Project Objectives

- Fetch and clean live US Treasury yield curve data from FRED.
- Apply PCA and interpret the first three components.
- Fit parametric interest rate models (Nelson-Siegel, Svensson, Vasicek, CIR).
- Correlate yield curve factors with macroeconomic indicators (GDP, CPI, Fed Funds Rate).
- Analyze yield curve behavior around key economic events.
- Forecast the yield curve 12 months ahead using a VAR(1) model.
- Visualize factor dynamics, rolling PCA, heatmaps, and animated yield curve evolution.

## Key Concepts

The **yield curve** plots interest rates of government bonds across maturities. PCA extracts dominant patterns:
- **PC1 (Level)**: Parallel shifts in rates (~88% of variance)
- **PC2 (Slope)**: Steepness — difference between short- and long-term rates (~11%)
- **PC3 (Curvature)**: Mid-term bending of the curve (~1%)

## Technologies Used

- Python 3
- pandas, numpy, scikit-learn, scipy, statsmodels
- matplotlib, seaborn

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Fetch data:**
```bash
python fetch_yield_data.py   # Downloads yield_curve_data.csv from FRED
python macro_data.py          # Downloads macro_data.csv from FRED
```

**2. Run the full analysis:**
```bash
python main.py
```

## Project Structure

```
├── main.py                    # Orchestrates the full analysis pipeline
├── fetch_yield_data.py        # Fetches US Treasury yield data from FRED
├── macro_data.py              # Fetches macroeconomic indicators from FRED
├── requirements.txt
├── src/
│   ├── data_loader.py         # Data loading and cleaning
│   ├── pca_model.py           # PCA, rolling PCA, t-SNE, statistical analysis
│   ├── parametric_models.py   # Nelson-Siegel, Svensson, Vasicek, CIR models
│   ├── macro_analysis.py      # Macro linkage analysis and event studies
│   ├── forecasting.py         # VAR(1) forecasting and yield curve reconstruction
│   └── visualization.py       # All plotting and animation functions
├── yield_curve_data.csv       # Fetched yield curve data
├── yield_curve_factors.csv    # PCA factor scores
└── macro_data.csv             # Macroeconomic indicators
```

## Outputs

| File | Description |
|------|-------------|
| `yield_curve_factors.csv` | Time series of PC1, PC2, PC3 factor scores |
| `forecasted_factors.csv` | 12-month VAR(1) factor forecasts |
| `forecasted_yield_curves.csv` | Reconstructed forecasted yield curves |
| `yield_curve_animation.gif` | Animated yield curve evolution over time |
