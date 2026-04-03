import numpy as np
import pandas as pd

beta = pd.read_csv('test11_2_beta.csv', header=0, index_col=0).values
factor_ret = pd.read_csv('test11_2_factor_returns.csv', header=0, index_col=None).values
stock_ret = pd.read_csv('test11_2_stock_returns.csv', header=0, index_col=None).values
w0 = pd.read_csv('test11_2_weights.csv', header=0, index_col=None).values.flatten()

T, m = factor_ret.shape
n = len(w0)

# Residuals: e_i,t = r_i,t - beta_i @ F_t
resid = stock_ret - factor_ret @ beta.T  # TxN

# Stock weights through time
W_s = np.zeros((T, n))
W_s[0] = w0
for t in range(T):
    w_star = W_s[t] * (1 + stock_ret[t])
    if t < T - 1:
        W_s[t+1] = w_star / w_star.sum()

# Factor weights each period: w_j,t = sum_i w_i,t * beta_i,j
W_f = W_s @ beta  # TxM

# Portfolio return and alpha per period
R = np.array([(W_s[t] * (1 + stock_ret[t])).sum() - 1 for t in range(T)])
alpha = np.array([W_s[t] @ resid[t] for t in range(T)])

# Carino K
total_R = np.prod(1 + R) - 1
GR = np.log(1 + total_R)
K = GR / total_R
k = np.log(1 + R) / (K * R)

# Return attribution
A_f = np.zeros(m)
for t in range(T):
    A_f += k[t] * W_f[t] * factor_ret[t]
A_alpha = np.sum(k * alpha)

# Risk attribution
port_ret = R
RA_f = np.zeros(m)
for j in range(m):
    wr = np.array([W_f[t, j] * factor_ret[t, j] for t in range(T)])
    beta_reg = np.cov(wr, port_ret)[0, 1] / np.var(port_ret)
    RA_f[j] = beta_reg * np.std(port_ret)
RA_alpha = np.cov(alpha, port_ret)[0, 1] / np.var(port_ret) * np.std(port_ret)

factor_total = np.prod(1 + factor_ret, axis=0) - 1
alpha_total = np.prod(1 + alpha) - 1

cols = ['F1', 'F2', 'F3', 'Alpha', 'Portfolio']
out = pd.DataFrame(
    [np.append(factor_total, [alpha_total, total_R]),
     np.append(A_f, [A_alpha, A_f.sum() + A_alpha]),
     np.append(RA_f, [RA_alpha, RA_f.sum() + RA_alpha])],
    index=['Total Return', 'Return Attribution', 'Vol Attribution'],
    columns=cols
)
out.index.name = 'Value'
out.to_csv('testout_11.2.csv')