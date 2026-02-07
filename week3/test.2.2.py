import pandas as pd
import numpy as np

lam = 0.94

df = pd.read_csv("test2.csv")   
X = df.to_numpy()
n, p = X.shape

# 1) w_{t-i} = (1-lam)*lam^(i-1), i=1..n --> w_i = (1-lam)*lam^(n-i),
i = np.arange(1, n + 1)
w = (1 - lam) * (lam ** (n - i))

# 2) Normalize weights so they sum to 1
w_hat = w / w.sum()

# 3) Exponentially weighted mean
mu = (w_hat.reshape(-1, 1) * X).sum(axis=0)

# 4) Exponentially weighted covariance matrix
Sigma = np.zeros((p, p))
for k in range(n):
    xt = X[k] - mu
    Sigma += w_hat[k] * np.outer(xt, xt)

# 5) Convert EW covariance to EW correlation: Corr_ij = Cov_ij / (sqrt(Cov_ii) * sqrt(Cov_jj))
var = np.diag(Sigma)
std = np.sqrt(var)
Corr = Sigma / np.outer(std, std)

Corr_df = pd.DataFrame(Corr, index=df.columns, columns=df.columns)
Corr_df.to_csv("testout_2.2.csv", index=False)