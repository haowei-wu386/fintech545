import numpy as np
import pandas as pd


# 1) Read the input covariance matrix Σ_in
df = pd.read_csv("test5_1.csv")
names = df.columns.tolist()
Sigma_in = df.to_numpy(dtype=float)

# 2) Cholesky factorization (works because Σ_in is PD) Σ = L L^T
L = np.linalg.cholesky(Sigma_in)

# 3) Set simulation size and random seed
n_sims = 100000
d = Sigma_in.shape[0]
seed=42
rng = np.random.default_rng(seed)  

# 4) Generate iid standard normals Z ~ N(0, I) Shape: (n_sims, d) so each row is one simulated observation
Z = rng.standard_normal((n_sims, d))

# 5) Construct samples X ~ N(0, Σ)
X_sim = Z @ L.T

# 6) Output covariance from simulated data (sample covariance, divide by N-1)
Sigma_out = np.cov(X_sim, rowvar=False, ddof=1)

# 7) Compare input vs output covariance
diff = Sigma_out - Sigma_in

abs_max = np.max(np.abs(diff))  # maximum absolute element difference
fro_in = np.linalg.norm(Sigma_in, ord="fro")
fro_diff = np.linalg.norm(diff, ord="fro")
rel_fro = fro_diff / (fro_in + 1e-18)

print("\n=== Error metrics (Output - Input) ===")
print(f"Max absolute diff           = {abs_max:.6e}")
print(f"Frobenius norm of diff      = {fro_diff:.6e}")
print(f"Relative Frobenius error    = {rel_fro:.6e}")

print("\n=== Input covariance (Sigma_in) ===")
print(pd.DataFrame(Sigma_in, index=names, columns=names))

print("\n=== Output covariance (Sigma_out) ===")
print(pd.DataFrame(Sigma_out, index=names, columns=names))

print("\n=== Difference (Sigma_out - Sigma_in) ===")
print(pd.DataFrame(diff, index=names, columns=names))

# 8) Save output covariance 
pd.DataFrame(Sigma_out, index=names, columns=names).to_csv("testout_5.1.csv", index=False)
