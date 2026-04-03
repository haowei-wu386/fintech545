import numpy as np
import pandas as pd

returns = pd.read_csv('test11_1_returns.csv', header=0, index_col=None).values
w0 = pd.read_csv('test11_1_weights.csv', header=0, index_col=None).values.flatten()

T, n = returns.shape

# Weights through time
W = np.zeros((T, n))
W[0] = w0
for t in range(T):
    w_star = W[t] * (1 + returns[t])
    if t < T - 1:
        W[t+1] = w_star / w_star.sum()

# Portfolio return each period
R = np.array([(W[t] * (1 + returns[t])).sum() - 1 for t in range(T)])

# Carino K
total_R = np.prod(1 + R) - 1
GR = np.log(1 + total_R)
K = GR / total_R
k = np.log(1 + R) / (K * R)

# Return attribution
A = np.zeros(n)
for t in range(T):
    A += k[t] * W[t] * returns[t]

# Risk attribution
RA = np.zeros(n)
port_ret = R
for i in range(n):
    wr = np.array([W[t, i] * returns[t, i] for t in range(T)])
    beta = np.cov(wr, port_ret)[0, 1] / np.var(port_ret)
    RA[i] = beta * np.std(port_ret)

stock_total = np.prod(1 + returns, axis=0) - 1

cols = [f'x{i+1}' for i in range(n)] + ['Portfolio']
out = pd.DataFrame(
    [np.append(stock_total, total_R),
     np.append(A, A.sum()),
     np.append(RA, RA.sum())],
    index=['Total Return', 'Return Attribution', 'Vol Attribution'],
    columns=cols
)

out.index.name = 'Value'
out.to_csv('testout_11.1.csv')