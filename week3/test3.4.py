import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.4.csv") 
# load correlation
C = df.to_numpy(dtype=float)
names = df.columns
n = C.shape[0]
C = 0.5 * (C + C.T) # Symmetrize

np.fill_diagonal(C, 1.0)

# Higham Algorithm on correlation (W = I) 
max_iter = 1000
tol = 1e-12
eps_eig = 0.0  # PSD

delta_S = np.zeros_like(C)
Y = C.copy()

for k in range(max_iter):
    Y_prev = Y.copy()

    R = Y - delta_S

    # P_S(R): project to PSD cone via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(0.5 * (R + R.T))
    eigvals = np.maximum(eigvals, eps_eig)
    X = (eigvecs * eigvals) @ eigvecs.T
    X = 0.5 * (X + X.T)

    # Update delta_S
    delta_S = X - R

    # P_U(X): enforce unit diagonal (correlation constraint)
    Y = X.copy()
    np.fill_diagonal(Y, 1.0)

    # Convergence check (Frobenius norm)
    if np.linalg.norm(Y - Y_prev, ord="fro") < tol:
        break

C_higham = 0.5 * (Y + Y.T)
np.fill_diagonal(C_higham, 1.0)

out_df = pd.DataFrame(C_higham, columns=names)
out_df.to_csv("testout_3.4.csv", index=False)
