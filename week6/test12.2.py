import pandas as pd
import numpy as np

# American option with continuous dividend yield
def bt_american(option_type, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)

    z = 1 if option_type == "Call" else -1

    def nNodeFunc(n):
        return (n + 1) * (n + 2) // 2

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)
    optionValues = np.zeros(nNodes)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            optionValues[idx] = max(0.0, z * (price - strike))

            if j < N:
                hold = np.exp(-rf * dt) * (
                    pu * optionValues[idxFunc(i + 1, j + 1)] +
                    pd * optionValues[idxFunc(i, j + 1)]
                )
                optionValues[idx] = max(optionValues[idx], hold)

    return optionValues[0]


# Value and Delta from the tree
def bt_american_with_delta(option_type, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)

    z = 1 if option_type == "Call" else -1

    def nNodeFunc(n):
        return (n + 1) * (n + 2) // 2

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)
    optionValues = np.zeros(nNodes)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            optionValues[idx] = max(0.0, z * (price - strike))

            if j < N:
                hold = df * (
                    pu * optionValues[idxFunc(i + 1, j + 1)] +
                    pd * optionValues[idxFunc(i, j + 1)]
                )
                optionValues[idx] = max(optionValues[idx], hold)

    value = optionValues[0]

    Vd = optionValues[idxFunc(0, 1)]
    Vu = optionValues[idxFunc(1, 1)]
    Sd = underlying * d
    Su = underlying * u
    delta = (Vu - Vd) / (Su - Sd)

    return value, delta


# Read input file
df = pd.read_csv("test12_1.csv")

# Output columns
df["Value"] = 0.0
df["Delta"] = 0.0
df["Gamma"] = 0.0
df["Vega"] = 0.0
df["Rho"] = 0.0
df["Theta"] = 0.0

N = 500

for i in range(len(df)):
    underlying = df.loc[i, "Underlying"]
    strike = df.loc[i, "Strike"]
    ttm = df.loc[i, "DaysToMaturity"] / df.loc[i, "DayPerYear"]
    rf = df.loc[i, "RiskFreeRate"]
    q = df.loc[i, "DividendRate"]
    ivol = df.loc[i, "ImpliedVol"]
    option_type = df.loc[i, "Option Type"]

    b = rf - q

    # Value and Delta
    value, delta = bt_american_with_delta(
        option_type, underlying, strike, ttm, rf, b, ivol, N
    )

    # Gamma
    dS = 1.5
    value_up = bt_american(option_type, underlying + dS, strike, ttm, rf, b, ivol, N)
    value_down = bt_american(option_type, underlying - dS, strike, ttm, rf, b, ivol, N)
    gamma = (value_up - 2 * value + value_down) / (dS ** 2)

    # Vega
    dV = 0.001
    value_up = bt_american(option_type, underlying, strike, ttm, rf, b, ivol + dV, N)
    value_down = bt_american(option_type, underlying, strike, ttm, rf, b, ivol - dV, N)
    vega = (value_up - value_down) / (2 * dV)

    # Rho (fix b)
    dR = 0.001
    value_up = bt_american(option_type, underlying, strike, ttm, rf + dR, b, ivol, N)
    value_down = bt_american(option_type, underlying, strike, ttm, rf - dR, b, ivol, N)
    rho = (value_up - value_down) / (2 * dR)

    # Theta
    dT = 1 / 365
    value_up = bt_american(option_type, underlying, strike, ttm + dT, rf, b, ivol, N)
    value_down = bt_american(option_type, underlying, strike, max(ttm - dT, 1e-8), rf, b, ivol, N)
    theta = (value_up - value_down) / (2 * dT)

    df.loc[i, "Value"] = value
    df.loc[i, "Delta"] = delta
    df.loc[i, "Gamma"] = gamma
    df.loc[i, "Vega"] = vega
    df.loc[i, "Rho"] = rho
    df.loc[i, "Theta"] = theta

out = df[["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]].copy()
out = out.dropna()
out["ID"] = out["ID"].astype(int)

out.to_csv("testout_12.2.csv", index=False)