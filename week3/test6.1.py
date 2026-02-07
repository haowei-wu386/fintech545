import pandas as pd

df = pd.read_csv("test6.csv")  
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# r_t = (P_t - P_{t-1}) / P_{t-1}
returns = df.pct_change()
returns = returns.dropna()

returns.to_csv("testout_6.1.csv")
