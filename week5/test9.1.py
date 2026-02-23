import numpy as np
import pandas as pd
from scipy.stats import norm, t

# Settings
alpha = 0.95
n_sims = 200000
seed = 42

portfolio = pd.read_csv("test9_1_portfolio.csv").set_index("Stock")
rets = pd.read_csv("test9_1_returns.csv")

# Dollar exposure (position value)
pv = portfolio["Holding"] * portfolio["Starting Price"]
stocks = list(pv.index)

# Returns matrix (T x d)
R = rets[stocks].dropna()
T, d = R.shape

# Fit marginals + build CDF / PPF functions
def fit_cdf_ppf(x: np.ndarray, dist_name: str):
    """
    Return (cdf_fn, ppf_fn, fitted_params_dict)
    """
    dist = str(dist_name).strip().lower()

    if dist.startswith("n"):  # Normal
        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1))

        def cdf_fn(v):
            return norm.cdf(v, loc=mu, scale=sigma)

        def ppf_fn(u):
            return norm.ppf(u, loc=mu, scale=sigma)

        return cdf_fn, ppf_fn, {"dist": "Normal", "mu": mu, "sigma": sigma}

    if dist.startswith("t"):  # Student-t
        nu, loc, scale = t.fit(x)

        def cdf_fn(v):
            return t.cdf(v, df=nu, loc=loc, scale=scale)

        def ppf_fn(u):
            return t.ppf(u, df=nu, loc=loc, scale=scale)

        return cdf_fn, ppf_fn, {"dist": "StudentT", "nu": float(nu), "loc": float(loc), "scale": float(scale)}

    raise ValueError(f"Unknown Distribution: {dist_name}")


cdf_fns = {}
ppf_fns = {}
fitted_params = {}

for s in stocks:
    x = R[s].to_numpy(dtype=float)
    dist_name = portfolio.loc[s, "Distribution"]
    cdf_fn, ppf_fn, params = fit_cdf_ppf(x, dist_name)
    cdf_fns[s] = cdf_fn
    ppf_fns[s] = ppf_fn
    fitted_params[s] = params

# Step 1: X -> U using fitted CDFs
eps = 1e-10  # avoid exactly 0 or 1
U = np.zeros((T, d), dtype=float)

for j, s in enumerate(stocks):
    u = cdf_fns[s](R[s].to_numpy(dtype=float))
    U[:, j] = np.clip(u, eps, 1 - eps)

# Step 2: Spearman correlation (robust) on U
# (Because Spearman uses ranks + monotonic transforms, computing on U is OK.)
U_df = pd.DataFrame(U, columns=stocks)
rho_s = U_df.corr(method="spearman").to_numpy()

# small jitter for numerical stability
rho_s = rho_s + np.eye(d) * 1e-12

# Step 3: Simulate Gaussian copula
# Z_sim ~ N(0, rho_s) -> U_sim = Phi(Z_sim)
rng = np.random.default_rng(seed)
Z_sim = rng.multivariate_normal(mean=np.zeros(d), cov=rho_s, size=n_sims)
U_sim = norm.cdf(Z_sim)
U_sim = np.clip(U_sim, eps, 1 - eps)

# Step 4: U -> X using fitted PPFs (inverse CDF)
R_sim = np.zeros((n_sims, d), dtype=float)
for j, s in enumerate(stocks):
    R_sim[:, j] = ppf_fns[s](U_sim[:, j])

# Step 5: PnL / Loss
pv_vec = pv.loc[stocks].to_numpy(dtype=float)  # (d,)
pnl_sim = R_sim * pv_vec                       # (n_sims, d)
loss_sim = -pnl_sim

loss_total = -pnl_sim.sum(axis=1)

# VaR / ES
def var_es(losses, alpha=0.95):
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean()
    return var, es

rows = []

# Single assets
for j, s in enumerate(stocks):
    var95, es95 = var_es(loss_sim[:, j], alpha)
    rows.append({
        "Stock": s,
        "VaR95": var95,
        "ES95": es95,
        "VaR95_Pct": var95 / float(pv[s]),
        "ES95_Pct": es95 / float(pv[s]),
    })

# Portfolio
var95_t, es95_t = var_es(loss_total, alpha)
pv_total = float(pv.sum())

rows.append({
    "Stock": "Total",
    "VaR95": var95_t,
    "ES95": es95_t,
    "VaR95_Pct": var95_t / pv_total,
    "ES95_Pct": es95_t / pv_total
})

out = pd.DataFrame(rows)
out.to_csv("testout9_1.csv", index=False)