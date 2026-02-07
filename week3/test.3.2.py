import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.4.csv")  
# load correlation 
C = df.to_numpy(dtype=float)
names = df.columns
n = C.shape[0]
C = 0.5 * (C + C.T) # enforce symmetry

# 1) Eigenvalue decomposition: C S = S Λ
eigvals, S = np.linalg.eigh(C)

# 2) Clip eigenvalues: λ'_i = max(λ_i, 0)
eigvals_clipped = np.maximum(eigvals, 0.0)

# 3) Construct diagonal scaling matrix T; t_i = [ sum_{j=1}^n s_{i,j}^2 * λ'_j ]^{-1}
t = np.zeros(n)
for i in range(n):
    t[i] = 1.0 / np.sum((S[i, :] ** 2) * eigvals_clipped)

sqrt_T = np.diag(np.sqrt(t))
sqrt_Lambda = np.diag(np.sqrt(eigvals_clipped))

# 4) B = sqrt(T) S sqrt(Λ'), and C_hat = B B^T
B = sqrt_T @ S @ sqrt_Lambda
C_psd = B @ B.T
C_psd = 0.5 * (C_psd + C_psd.T)

out_df = pd.DataFrame(C_psd, columns=names)
out_df.to_csv("testout_3.2.csv", index=False)


