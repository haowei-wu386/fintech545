import pandas as pd
from scipy.stats import norm

df = pd.read_csv("test7_1.csv")
x1 = df["x1"]

mu = x1.mean()
sigma = x1.std(ddof=1) 
z = norm.ppf(0.95)

var_diff_mean = z * sigma
var_absolute = var_diff_mean - mu


out = pd.DataFrame({
    "VaR Absolute": [var_absolute],
    "VaR Diff from Mean": [var_diff_mean]
})
out.to_csv("testout_8.1.csv", index=False)