import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv('test5_2.csv', header=0, index_col=None)
Sigma = df.values  # 5x5
n = Sigma.shape[0]

def CSD(w):
    return w * (Sigma @ w) / np.sqrt(w @ Sigma @ w)

def SSE(w):
    csd = CSD(w)
    return np.sum((csd - csd.mean()) ** 2)

w0 = 1 / np.sqrt(np.diag(Sigma))
w0 = w0 / w0.sum()

res = minimize(SSE, w0, method='SLSQP',
               bounds=[(0, None)] * n,
               constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
               options={'ftol': 1e-12, 'maxiter': 1000})

pd.DataFrame({'W': res.x}).to_csv('testout_10.1.csv', index=False)