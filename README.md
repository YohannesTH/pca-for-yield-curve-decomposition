# PCA for Yield Curve Decomposition

This project demonstrates how **Principal Component Analysis (PCA)** can be applied to decompose the **yield curve** into its main components—commonly interpreted as **level**, **slope**, and **curvature**. The method is widely used in fixed-income research to understand and model interest rate movements.

## Project Objectives

- Load and clean historical yield curve data.
- Apply PCA to the yield curve across different maturities.
- Visualize and interpret the first few principal components.
- Analyze how these components capture the dynamics of interest rates.

## What is the Yield Curve?

The yield curve plots interest rates of bonds with the same credit quality but different maturities. It reflects market expectations of future interest rates, inflation, and economic activity. PCA helps extract dominant patterns from these curves.

## Why PCA?

PCA allows us to reduce the dimensionality of the yield curve data while retaining most of the variation. The first three components typically capture:
- **Level**: Overall shift in interest rates.
- **Slope**: Difference between short-term and long-term rates.
- **Curvature**: Bending of the curve (mid-term behavior).

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Requests

## Setup Instructions

1. **Clone the repository and navigate to the project directory.**
2. (Optional but recommended) Create a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

- To download the latest US Treasury yield curve data, run:
  ```bash
  python fetch_yield_data.py
  ```
  This will create `yield_curve_data.csv` in the project directory.

- To download macroeconomic data (GDP growth, CPI, Fed Funds Rate), run:
  ```bash
  python macro_data.py
  ```
  This will create `macro_data.csv` in the project directory.

## Running the Analysis

The main analysis script is `pca_yield_curve.py`. To run the full PCA decomposition and related analyses:

```bash
python pca_yield_curve.py
```

This will:
- Load and clean the yield curve data
- Perform PCA and visualize the main components
- Save the factor time series to `yield_curve_factors.csv`
- Fit and compare Nelson-Siegel, Svensson, Vasicek, and CIR models
- Analyze macroeconomic linkages
- Generate various plots and visualizations

## Main Files

- `fetch_yield_data.py`: Downloads US Treasury yield curve data from FRED
- `macro_data.py`: Downloads macroeconomic data from FRED
- `pca_yield_curve.py`: Main analysis script (PCA, modeling, visualization)
- `yield_curve_data.csv`: Yield curve data (generated)
- `macro_data.csv`: Macroeconomic data (generated)
- `yield_curve_factors.csv`: PCA factor time series (generated)

---

Feel free to open issues or contribute if you have suggestions or improvements!

