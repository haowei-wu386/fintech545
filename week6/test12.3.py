import pandas as pd
import numpy as np


def bt_american(option_type, underlying, strike, ttm, rf, b, ivol, N):
    if N <= 0 or ttm <= 0:
        if option_type == "Call":
            return max(underlying - strike, 0.0)
        else:
            return max(strike - underlying, 0.0)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)

    z = 1 if option_type == "Call" else -1

    values = np.zeros(N + 1)
    for i in range(N + 1):
        price = underlying * (u ** i) * (d ** (N - i))
        values[i] = max(0.0, z * (price - strike))

    for j in range(N - 1, -1, -1):
        new_values = np.zeros(j + 1)
        for i in range(j + 1):
            price = underlying * (u ** i) * (d ** (j - i))
            exercise = max(0.0, z * (price - strike))
            hold = df * (pu * values[i + 1] + pd * values[i])
            new_values[i] = max(exercise, hold)
        values = new_values

    return values[0]


def bt_american_div(option_type, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N):
    if N <= 0 or ttm <= 0:
        if option_type == "Call":
            return max(underlying - strike, 0.0)
        else:
            return max(strike - underlying, 0.0)

    if len(divAmts) == 0 or len(divTimes) == 0:
        return bt_american(option_type, underlying, strike, ttm, rf, rf, ivol, N)

    if divTimes[0] > N:
        return bt_american(option_type, underlying, strike, ttm, rf, rf, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)

    z = 1 if option_type == "Call" else -1
    firstDivTime = divTimes[0]

    values = np.zeros(firstDivTime + 1)

    for i in range(firstDivTime + 1):
        price = underlying * (u ** i) * (d ** (firstDivTime - i))

        new_underlying = price - divAmts[0]
        new_ttm = ttm - firstDivTime * dt
        new_divAmts = divAmts[1:]
        new_divTimes = [t - firstDivTime for t in divTimes[1:]]
        new_N = N - firstDivTime

        val_no_exercise = bt_american_div(
            option_type,
            new_underlying,
            strike,
            new_ttm,
            rf,
            new_divAmts,
            new_divTimes,
            ivol,
            new_N
        )

        val_exercise = max(0.0, z * (price - strike))
        values[i] = max(val_no_exercise, val_exercise)

    for j in range(firstDivTime - 1, -1, -1):
        new_values = np.zeros(j + 1)
        for i in range(j + 1):
            price = underlying * (u ** i) * (d ** (j - i))
            exercise = max(0.0, z * (price - strike))
            hold = df * (pu * values[i + 1] + pd * values[i])
            new_values[i] = max(exercise, hold)
        values = new_values

    return values[0]


df = pd.read_csv("test12_3.csv")
df["Value"] = 0.0

for i in range(len(df)):
    option_type = df.loc[i, "Option Type"]
    underlying = float(df.loc[i, "Underlying"])
    strike = float(df.loc[i, "Strike"])
    days = int(df.loc[i, "DaysToMaturity"])
    day_per_year = float(df.loc[i, "DayPerYear"])
    ttm = days / day_per_year
    rf = float(df.loc[i, "RiskFreeRate"])
    ivol = float(df.loc[i, "ImpliedVol"])

    divDatesStr = str(df.loc[i, "DividendDates"]).strip()
    divAmtsStr = str(df.loc[i, "DividendAmts"]).strip()

    divDates = [int(x.strip()) for x in divDatesStr.split(",") if x.strip() != ""]
    divAmts = [float(x.strip()) for x in divAmtsStr.split(",") if x.strip() != ""]

    # use one step per day
    N = days

    # map dividend dates onto the tree grid
    divTimes = [round(d / days * N) for d in divDates]

    value = bt_american_div(
        option_type,
        underlying,
        strike,
        ttm,
        rf,
        divAmts,
        divTimes,
        ivol,
        N
    )

    df.loc[i, "Value"] = value

out = df[["ID", "Value"]].copy()
out = out.dropna()
out["ID"] = out["ID"].astype(int)

out.to_csv("testout_12.3.csv", index=False)