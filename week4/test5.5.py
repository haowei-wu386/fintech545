import numpy as np
import pandas as pd


# 1) Read PSD covariance Σ_in
df = pd.read_csv("test5_2.csv")
names = df.columns.tolist()
Sigma_in = df.to_numpy(dtype=float)
Sigma_in = 0.5 * (Sigma_in + Sigma_in.T)  # enforce symmetry
n = Sigma_in.shape[0]

# 2) PCA / eigen-decomposition of Σ: Σ = S Λ S^T (sorted by descending eigenvalues)
eigvals, S = np.linalg.eigh(Sigma_in)

# sort eigenvalues/vectors descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
S = S[:, idx]

# clip tiny negatives due to rounding (PSD should be >=0)
eigvals = np.maximum(eigvals, 0.0)

# 3) Choose k components to explain 99% variance
total_var = eigvals.sum()
cum_var = np.cumsum(eigvals)
explained_ratio = cum_var / (total_var + 1e-18)

pctExp = 0.99
k = int(np.searchsorted(explained_ratio, pctExp) + 1)

print("Eigenvalues (desc):", eigvals)
print("Total variance:", total_var)
print("Chosen pctExp:", pctExp)
print("Chosen k:", k)
print("Explained variance at k:", explained_ratio[k - 1])


# 4) Build PCA root A_k = S_k * sqrt(Λ_k) so that Σ ≈ A_k A_k^T
S_k = S[:, :k]                      # (n x k)
Lambda_k_sqrt = np.diag(np.sqrt(eigvals[:k]))   # (k x k)
A_k = S_k @ Lambda_k_sqrt           # (n x k)
Sigma_recon = A_k @ A_k.T


# 5) Simulate: Z ~ N(0, I_k), X = (A_k Z)^T and keep mean = 0
N = 100000
seed = 42 
rng = np.random.default_rng(seed)

Z = rng.standard_normal((k, N))     # (k x N)
X_sim = (A_k @ Z).T                 # (N x n)


# 6) Output covariance from simulation
Sigma_out = np.cov(X_sim, rowvar=False, ddof=1)


# 7) Compare output vs input covariance
#    (Note: if k < n, Σ is approximated; so Sigma_out should match the PCA-approx Σ_recon very well,
#     and be close to Sigma_in up to the truncation error.)
diff_in = Sigma_out - Sigma_in
abs_max_in = np.max(np.abs(diff_in))
fro_in = np.linalg.norm(Sigma_in, ord="fro")
fro_diff_in = np.linalg.norm(diff_in, ord="fro")
rel_fro_in = fro_diff_in / (fro_in + 1e-18)

diff_recon = Sigma_out - Sigma_recon
abs_max_recon = np.max(np.abs(diff_recon))
fro_recon = np.linalg.norm(Sigma_recon, ord="fro")
fro_diff_recon = np.linalg.norm(diff_recon, ord="fro")
rel_fro_recon = fro_diff_recon / (fro_recon + 1e-18)

print("\n=== Error metrics vs original input (Sigma_out - Sigma_in) ===")
print(f"Max absolute diff           = {abs_max_in:.6e}")
print(f"Frobenius norm of diff      = {fro_diff_in:.6e}")
print(f"Relative Frobenius error    = {rel_fro_in:.6e}")

print("\n=== Error metrics vs PCA approximation (Sigma_out - Sigma_recon) ===")
print(f"Max absolute diff           = {abs_max_recon:.6e}")
print(f"Frobenius norm of diff      = {fro_diff_recon:.6e}")
print(f"Relative Frobenius error    = {rel_fro_recon:.6e}")

print("\n=== Input covariance (Sigma_in) ===")
print(pd.DataFrame(Sigma_in, index=names, columns=names))

print("\n=== Output covariance (Sigma_out) ===")
print(pd.DataFrame(Sigma_out, index=names, columns=names))


# 8) Save output covariance 
pd.DataFrame(Sigma_out, index=names, columns=names).to_csv("testout_5.5.csv", index=False)
