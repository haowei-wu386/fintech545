import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.3.csv")
# load covariance
Sigma = df.to_numpy(dtype=float)
names = df.columns
n = Sigma.shape[0]

# 1) convert covariance -> correlation
std = np.sqrt(np.diag(Sigma))
D_inv = np.diag(1.0 / std)
C = D_inv @ Sigma @ D_inv
C = 0.5 * (C + C.T)   # enforce symmetry

# 2) spectral decomposition of correlation: C S = S Λ
eigvals, S = np.linalg.eigh(C)

# 3) clip eigenvalues:  λ'_i = max(λ_i, 0)
eigvals_clipped = np.maximum(eigvals, 0.0)
Lambda_p = np.diag(eigvals_clipped)

# 4) Construct diagonal scaling matrix T; t_i = [ sum_{j=1}^n s_{i,j}^2 * λ'_j ]^{-1}
t = np.zeros(n)
for i in range(n):
    t[i] = 1.0 / np.sum((S[i, :] ** 2) * eigvals_clipped)

T = np.diag(t)

# 5) reconstruct PSD correlation
sqrt_T = np.diag(np.sqrt(t))
sqrt_Lambda = np.diag(np.sqrt(eigvals_clipped))

B = sqrt_T @ S @ sqrt_Lambda
C_psd = B @ B.T
C_psd = 0.5 * (C_psd + C_psd.T)

# 6) convert back to covariance 
D = np.diag(std)
Sigma_psd = D @ C_psd @ D

out_df = pd.DataFrame(Sigma_psd, columns=names)
out_df.to_csv("testout_3.1.csv", index=False)

