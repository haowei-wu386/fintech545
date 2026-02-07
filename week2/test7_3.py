import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize

df = pd.read_csv("test7_3.csv")
X = df[["x1", "x2", "x3"]].to_numpy()
y = df["y"].to_numpy()


def neg_loglik(theta):
    alpha = theta[0]
    b1 = theta[1]
    b2 = theta[2]
    b3 = theta[3]
    mu = theta[4]
    sigma = np.exp(theta[5])   # sigma >0
    nu = np.exp(theta[6])      # nu >0

    resid = y - (alpha + b1*X[:,0] + b2*X[:,1] + b3*X[:,2])

    return -np.sum(t.logpdf(resid, df=nu, loc=mu, scale=sigma))


# initial guess
theta0 = [
    0.0,    # alpha
    0.0,    # b1
    0.0,    # b2
    0.0,    # b3
    0.0,    # mu
    np.log(np.std(y)),  # sigma
    np.log(5.0)         # nu
]

res = minimize(neg_loglik, theta0, method="L-BFGS-B")

# result
Alpha = res.x[0]
B1    = res.x[1]
B2    = res.x[2]
B3    = res.x[3]
mu    = res.x[4]
sigma = np.exp(res.x[5])
nu    = np.exp(res.x[6])

pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma],
    "nu": [nu],
    "Alpha": [Alpha],
    "B1": [B1],
    "B2": [B2],
    "B3": [B3]
}).to_csv("testout7_3.csv", index=False)
