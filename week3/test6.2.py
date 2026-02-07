import pandas as pd
import numpy as np

df = pd.read_csv("test6.csv") 
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# r_t = ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})
log_returns = np.log(df / df.shift(1))
log_returns = log_returns.dropna()

log_returns.to_csv("testout_6.2.csv")
