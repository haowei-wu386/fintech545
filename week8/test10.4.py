import numpy as np
import pandas as pd
from scipy.optimize import minimize

Sigma = pd.read_csv('test5_2.csv', header=0, index_col=None).values
mu = pd.read_csv('test10_3_means.csv', header=0, index_col=None).values.flatten()
rf = 0.04
n = len(mu)

def neg_sharpe(w):
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    return -(ret - rf) / vol

w0 = np.ones(n) / n

res = minimize(neg_sharpe, w0, method='SLSQP',
               bounds=[(0.1, 0.5)] * n,
               constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
               options={'ftol': 1e-12, 'maxiter': 1000})

pd.DataFrame({'W': res.x}).to_csv('testout_10.4.csv', index=False)