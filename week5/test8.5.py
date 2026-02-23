import pandas as pd
from scipy.stats import t

alpha = 0.95 

df = pd.read_csv("test7_2.csv")
x = df["x1"]

# Convert returns to losses
loss = -x

# Fit t distribution
nu, mu, sigma = t.fit(loss)

# Compute the right-tail quantile
t_alpha = t.ppf(alpha, df=nu)

# Evaluate t PDF at the quantile
pdf_val = t.pdf(t_alpha, df=nu)

# Closed-form Expected Shortfall (ES) for Student-t (right tail)
es_raw = mu + sigma * (pdf_val * (nu + t_alpha**2) / ((1 - alpha) * (nu - 1)))

ES_absolute = es_raw
ES_diff_from_mean = es_raw - loss.mean()

result = pd.DataFrame({
    "ES Absolute": [ES_absolute],
    "ES Diff from Mean": [ES_diff_from_mean]
})

result.to_csv("testout8_5.csv", index=False)
