import pandas as pd
import numpy as np
from scipy.stats import norm

# Read input file
df = pd.read_csv("test12_1.csv")

# Create output columns
df["Value"] = 0.0
df["Delta"] = 0.0
df["Gamma"] = 0.0
df["Vega"] = 0.0
df["Rho"] = 0.0
df["Theta"] = 0.0

# Loop through each row
for i in range(len(df)):
    S = df.loc[i, "Underlying"]
    X = df.loc[i, "Strike"]
    T = df.loc[i, "DaysToMaturity"] / df.loc[i, "DayPerYear"]
    r = df.loc[i, "RiskFreeRate"]
    q = df.loc[i, "DividendRate"]
    sigma = df.loc[i, "ImpliedVol"]
    option_type = df.loc[i, "Option Type"]

    # Calculate d1 and d2
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Call option
    if option_type == "Call":
        value = S * np.exp(-q * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        rho = X * T * np.exp(-r * T) * norm.cdf(d2)
        theta = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + q * S * np.exp(-q * T) * norm.cdf(d1)
            - r * X * np.exp(-r * T) * norm.cdf(d2)
        )

    # Put option
    else:
        value = X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        rho = -X * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
            + r * X * np.exp(-r * T) * norm.cdf(-d2)
        )

    # Greeks for both call and put
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    # Save results
    df.loc[i, "Value"] = value
    df.loc[i, "Delta"] = delta
    df.loc[i, "Gamma"] = gamma
    df.loc[i, "Vega"] = vega
    df.loc[i, "Rho"] = rho
    df.loc[i, "Theta"] = theta

# Keep only required columns
out = df[["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]].copy()

# Remove empty rows and fix ID format
out = out.dropna()
out["ID"] = out["ID"].astype(int)

# Export result
out.to_csv("testout_12.1.csv", index=False)