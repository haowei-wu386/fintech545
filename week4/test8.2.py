import pandas as pd
from scipy.stats import t

df0 = pd.read_csv("test7_2.csv")
x1 = df0["x1"].dropna()

df_hat, loc_hat, scale_hat = t.fit(x1)

q05 = t.ppf(0.05, df_hat) 
var_diff_mean = -q05 * scale_hat
var_absolute = var_diff_mean - loc_hat

out = pd.DataFrame({
    "VaR Absolute": [var_absolute],
    "VaR Diff from Mean": [var_diff_mean]
})
out.to_csv("testout_8.2.csv", index=False)
