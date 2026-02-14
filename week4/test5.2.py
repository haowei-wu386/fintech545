import numpy as np
import pandas as pd

# 1) Read PSD covariance matrix
df = pd.read_csv("test5_2.csv")
names = df.columns.tolist()
Sigma_in = df.to_numpy(dtype=float)
n = Sigma_in.shape[0]

Sigma_in = 0.5 * (Sigma_in + Sigma_in.T) # Ensure symmetry

# 2) chol_psd: build lower-triangular L such that Sigma_in ≈ L L^T (PSD allowed)
L = np.zeros((n, n), dtype=float)
tol = 1e-12

for j in range(n):
    # compute pivot = Sigma[j,j] - sum_{k<j} L[j,k]^2
    s = 0.0
    if j > 0:
        s = np.dot(L[j, :j], L[j, :j])

    pivot = Sigma_in[j, j] - s

    # numerical PSD handling: tiny negative -> treat as 0
    if pivot < 0 and abs(pivot) < tol:
        pivot = 0.0

    # if strongly negative, it's not PSD (this shouldn't happen for test5_2)
    if pivot < 0:
        raise ValueError(f"Matrix is not PSD at j={j}, pivot={pivot}")

    L[j, j] = np.sqrt(pivot)

    # fill below diagonal
    if L[j, j] > tol:
        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (Sigma_in[i, j] - s) / L[j, j]
    else:
        # PSD case: diagonal ~0 => set column entries to 0 to avoid divide-by-0
        for i in range(j + 1, n):
            L[i, j] = 0.0

# 3) Simulate Z ~ N(0, I)
N = 100000
seed = 42
rng = np.random.default_rng(seed)
Z = rng.standard_normal((n, N))

# 4) Create simulated samples X_sim ~ N(0, Sigma_in)
X_sim = (L @ Z).T   # shape (N, n), rows=samples, cols=variables

# 5) Compute output covariance (unbiased: divide by N-1)
Sigma_out = np.cov(X_sim, rowvar=False, ddof=1)

# 6) Compare input vs output covariance
diff = Sigma_out - Sigma_in
abs_max = np.max(np.abs(diff))
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

# 7) Save output covariance 
pd.DataFrame(Sigma_out, index=names, columns=names).to_csv("testout_5.2.csv", index=False)
