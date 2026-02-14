import numpy as np
import pandas as pd

# 1) Read nonPSD covariance Σ_in
df = pd.read_csv("test5_3.csv")
names = df.columns.tolist()
Sigma_in = df.to_numpy(dtype=float)
Sigma_in = 0.5 * (Sigma_in + Sigma_in.T)  # enforce symmetry
n = Sigma_in.shape[0]


# 2) near_psd fix  cov -> corr -> eigen clip -> rescale diag -> back to cov

# cov -> corr
std = np.sqrt(np.diag(Sigma_in))
D_inv = np.diag(1.0 / std)
C = D_inv @ Sigma_in @ D_inv
C = 0.5 * (C + C.T)

# eigendecomposition of correlation
eigvals, S = np.linalg.eigh(C)

# clip eigenvalues (negative -> 0)
eigvals_clipped = np.maximum(eigvals, 0.0)

# build T so that diag is re-normalized to 1
t = np.zeros(n)
for i in range(n):
    t[i] = 1.0 / np.sum((S[i, :] ** 2) * eigvals_clipped)

sqrt_T = np.diag(np.sqrt(t))
sqrt_Lambda = np.diag(np.sqrt(eigvals_clipped))

# reconstruct PSD correlation
B = sqrt_T @ S @ sqrt_Lambda
C_psd = B @ B.T
C_psd = 0.5 * (C_psd + C_psd.T)

# corr -> cov (keep original std)
D = np.diag(std)
Sigma_fixed = D @ C_psd @ D
Sigma_fixed = 0.5 * (Sigma_fixed + Sigma_fixed.T)


# 3) chol_psd (PSD-aware Cholesky root) on Sigma_fixed
#    Build L such that Sigma_fixed ≈ L L^T, allowing zero pivots

L = np.zeros((n, n), dtype=float)
tol = 1e-12

for j in range(n):
    s = 0.0
    if j > 0:
        s = np.dot(L[j, :j], L[j, :j])

    pivot = Sigma_fixed[j, j] - s

    # tiny negative due to rounding -> treat as 0
    if pivot < 0 and abs(pivot) < tol:
        pivot = 0.0

    if pivot < 0:
        raise ValueError(f"Sigma_fixed is not PSD at j={j}, pivot={pivot}")

    L[j, j] = np.sqrt(pivot)

    if L[j, j] > tol:
        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (Sigma_fixed[i, j] - s) / L[j, j]
    else:
        for i in range(j + 1, n):
            L[i, j] = 0.0


# 4) Normal simulation: mean=0, N=100,000

N = 100000
seed = 42
rng = np.random.default_rng(seed)

Z = rng.standard_normal((n, N))
X_sim = (L @ Z).T

# 5) Output covariance from simulation 
Sigma_out = np.cov(X_sim, rowvar=False, ddof=0)

# 6) Compare input vs output covariance
diff = Sigma_out - Sigma_fixed
abs_max = np.max(np.abs(diff))
fro_in = np.linalg.norm(Sigma_fixed, ord="fro")
fro_diff = np.linalg.norm(diff, ord="fro")
rel_fro = fro_diff / (fro_in + 1e-18)

print("\n=== Error metrics (Sigma_out - Sigma_fixed) ===")
print(f"Max absolute diff           = {abs_max:.6e}")
print(f"Frobenius norm of diff      = {fro_diff:.6e}")
print(f"Relative Frobenius error    = {rel_fro:.6e}")

print("\n=== Fixed covariance (Sigma_fixed) ===")
print(pd.DataFrame(Sigma_fixed, index=names, columns=names))

print("\n=== Output covariance (Sigma_out) ===")
print(pd.DataFrame(Sigma_out, index=names, columns=names))

# 7) Save output covariance 
pd.DataFrame(Sigma_out, index=names, columns=names).to_csv("testout_5.3.csv", index=False)

