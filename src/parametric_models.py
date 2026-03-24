import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def nelson_siegel(maturities, beta0, beta1, beta2, tau):
    maturities = np.array(maturities)
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-maturities / tau)) / (maturities / tau)
    term3 = beta2 * ((1 - np.exp(-maturities / tau)) / (maturities / tau) - np.exp(-maturities / tau))
    return term1 + term2 + term3

def fit_nelson_siegel_all(df):
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
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

def fit_vasicek_cir_all(df):
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
        try:
            popt, _ = curve_fit(lambda m, kappa, theta, sigma, r0: vasicek_yield(m, kappa, theta, sigma, r0),
                                maturities, row, p0=[0.1, 0.03, 0.01, row[0]],
                                bounds=([0.001, 0, 0, -1], [2, 0.2, 0.2, 1]))
        except Exception:
            popt = [np.nan]*4
        vasicek_params.append(popt)
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
