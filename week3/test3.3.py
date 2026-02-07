import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.3.csv")
# load covariance
Sigma = df.to_numpy(dtype=float)
names = df.columns
n = Sigma.shape[0]
Sigma = 0.5 * (Sigma + Sigma.T)

# 1) convert covariance -> correlation
std = np.sqrt(np.diag(Sigma))
D_inv = np.diag(1.0 / std)
C = D_inv @ Sigma @ D_inv
C = 0.5 * (C + C.T)

# 2) Higham Algorithm 3.3 on correlation (W = I)
max_iter = 1000
tol = 1e-12
eps_eig = 0.0  # PSD

delta_S = np.zeros_like(C)
Y = C.copy()

for k in range(max_iter):
    Y_prev = Y.copy()

    R = Y - delta_S

    # P_S(R): PSD projection via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(0.5 * (R + R.T))
    eigvals = np.maximum(eigvals, eps_eig)
    X = (eigvecs * eigvals) @ eigvecs.T
    X = 0.5 * (X + X.T)

    delta_S = X - R

    # P_U(X): enforce unit diagonal (correlation constraint)
    Y = X.copy()
    np.fill_diagonal(Y, 1.0)

    if np.linalg.norm(Y - Y_prev, ord="fro") < tol:
        break

C_higham = 0.5 * (Y + Y.T)
np.fill_diagonal(C_higham, 1.0)

# 6) convert back to covariance 
D = np.diag(std)
Sigma_higham = D @ C_higham @ D
Sigma_higham = 0.5 * (Sigma_higham + Sigma_higham.T)

out_df = pd.DataFrame(Sigma_higham, columns=names)
out_df.to_csv("testout_3.3.csv", index=False)


