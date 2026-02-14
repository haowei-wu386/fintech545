import numpy as np
import pandas as pd

# 1) Read nonPSD covariance Σ_in
df = pd.read_csv("test5_3.csv")
names = df.columns.tolist()
Sigma_in = df.to_numpy(dtype=float)
Sigma_in = 0.5 * (Sigma_in + Sigma_in.T)  # enforce symmetry
n = Sigma_in.shape[0]

# 2) Convert covariance -> correlation
std = np.sqrt(np.diag(Sigma_in))
D_inv = np.diag(1.0 / std)
C = D_inv @ Sigma_in @ D_inv
C = 0.5 * (C + C.T)

# 3) Higham nearest correlation (W = I): alternating projections
#    P_S: project to PSD (eigenvalue clipping)
#    P_U: enforce unit diagonal
max_iter = 1000
tol = 1e-12
eps_eig = 0.0  # PSD target (can keep 0; if you want strictly PD use small epsilon)

delta_S = np.zeros_like(C)
Y = C.copy()

for k in range(max_iter):
    Y_prev = Y.copy()

    R = Y - delta_S

    # P_S(R): PSD projection
    eigvals, eigvecs = np.linalg.eigh(0.5 * (R + R.T))
    eigvals = np.maximum(eigvals, eps_eig)
    X = (eigvecs * eigvals) @ eigvecs.T
    X = 0.5 * (X + X.T)

    delta_S = X - R

    # P_U(X): enforce diag = 1 (correlation constraint)
    Y = X.copy()
    np.fill_diagonal(Y, 1.0)

    if np.linalg.norm(Y - Y_prev, ord="fro") < tol:
        break

C_higham = 0.5 * (Y + Y.T)
np.fill_diagonal(C_higham, 1.0)

# 4) Convert correlation -> covariance (keep original std)
D = np.diag(std)
Sigma_fixed = D @ C_higham @ D
Sigma_fixed = 0.5 * (Sigma_fixed + Sigma_fixed.T)


# 5) chol_psd: PSD-aware Cholesky root of Sigma_fixed
L = np.zeros((n, n), dtype=float)
tol_chol = 1e-12

for j in range(n):
    s = 0.0
    if j > 0:
        s = np.dot(L[j, :j], L[j, :j])

    pivot = Sigma_fixed[j, j] - s

    if pivot < 0 and abs(pivot) < tol_chol:
        pivot = 0.0

    if pivot < 0:
        raise ValueError(f"Sigma_fixed is not PSD at j={j}, pivot={pivot}")

    L[j, j] = np.sqrt(pivot)

    if L[j, j] > tol_chol:
        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (Sigma_fixed[i, j] - s) / L[j, j]
    else:
        for i in range(j + 1, n):
            L[i, j] = 0.0


# 6) Normal simulation: mean=0, N=100,000
N = 100_000
seed = 42
rng = np.random.default_rng(seed)

Z = rng.standard_normal((n, N))
X_sim = (L @ Z).T


# 7) Output covariance and comparison
Sigma_out = np.cov(X_sim, rowvar=False, ddof=1)

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

# 8) Save output covariance 
pd.DataFrame(Sigma_out, index=names, columns=names).to_csv("testout_5.4.csv", index=False)

