import pandas as pd
from scipy.stats import norm

df = pd.read_csv("test7_1.csv")

alpha = 0.95

# Sample mean and sample standard deviation 
mu = df["x1"].mean()
sigma = df["x1"].std(ddof=1) 

# z-score at the alpha quantile
z = norm.ppf(alpha)

es_raw = mu - sigma * norm.pdf(z) / (1 - alpha)

es_absolute = abs(es_raw)
es_diff = abs(mu - es_raw)

result = pd.DataFrame({
    "ES Absolute": [es_absolute],
    "ES Diff from Mean": [es_diff]
})

result.to_csv("testout8_4.csv", index=False)