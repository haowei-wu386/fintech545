import numpy as np
import pandas as pd

# load near-PSD covariance
df = pd.read_csv("testout_3.1.csv")
Sigma = df.to_numpy(dtype=float)
names = df.columns
n = Sigma.shape[0]

Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize

# chol_psd (build lower-triangular L) 
L = np.zeros((n, n))
tol = 1e-12

for j in range(n):
    # diagonal element
    s = 0.0
    if j > 0:
        s = np.dot(L[j, :j], L[j, :j])

    pivot = Sigma[j, j] - s

    # PSD handling: if pivot is tiny negative, treat as 0
    if pivot < 0 and abs(pivot) < tol:
        pivot = 0.0

    if pivot < 0:
        raise ValueError(f"Matrix is not PSD enough at j={j}, pivot={pivot}")

    L[j, j] = np.sqrt(pivot)

    # off-diagonal elements below the diagonal
    if L[j, j] > tol:
        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (Sigma[i, j] - s) / L[j, j]
    else:
        # if diagonal is ~0, set column entries to 0 (PSD case)
        for i in range(j + 1, n):
            L[i, j] = 0.0

L_df = pd.DataFrame(L, index=names, columns=names)
L_df.to_csv("testout_4.1.csv", index=False)