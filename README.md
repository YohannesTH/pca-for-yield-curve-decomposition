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

## Folder Structure
├── data/ # Raw or processed yield curve data
├── notebooks/ # Jupyter notebooks for analysis and visualization
├── src/ # Python scripts for modular code
├── README.md

