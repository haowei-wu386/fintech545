import pandas as pd
import numpy as np

lam_var = 0.97 
lam_corr = 0.94  

df = pd.read_csv("test2.csv")
X = df.to_numpy()
n, p = X.shape

def ew_covariance(X, lam):
    n, p = X.shape

    # EW weights
    i = np.arange(1, n + 1)
    w = (1 - lam) * lam ** (n - i)
    w = w / w.sum()

    # EW mean
    mu = (w.reshape(-1, 1) * X).sum(axis=0)

    # EW covariance
    Sigma = np.zeros((p, p))
    for t in range(n):
        xt = X[t] - mu
        Sigma += w[t] * np.outer(xt, xt)

    return Sigma

def ew_correlation(X, lam):
    Sigma = ew_covariance(X, lam)
    std = np.sqrt(np.diag(Sigma))
    Corr = Sigma / np.outer(std, std)
    return Corr

# 1) EW variance / covariance (lambda = 0.97) 
Sigma_var = ew_covariance(X, lam_var)
var = np.diag(Sigma_var)

# 2) EW correlation (lambda = 0.94) 
Corr = ew_correlation(X, lam_corr)

# 3) combine 
Sigma_final = np.outer(np.sqrt(var), np.sqrt(var)) * Corr

Sigma_df = pd.DataFrame(Sigma_final, columns=df.columns, index=df.columns)
Sigma_df.to_csv("testout_2.3.csv", index=False)
