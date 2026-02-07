import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

x = pd.read_csv("test7_2.csv").iloc[:, 0].to_numpy()

def neg_loglik(theta):
    # [mu, log_sigma, log_nu] = theta
    mu = theta[0]
    sigma = np.exp(theta[1])  # sigma > 0
    nu = np.exp(theta[2])     # nu > 0
    
    return -np.sum(t.logpdf(x, nu, mu, sigma))

# initial guess
theta0 = [
    np.mean(x),        # mu is initialized at the sample mean
    np.log(np.std(x)), # sigma is initialized at the sample standard deviation
    np.log(10)         # nu is initially set to 10
]

res = minimize(neg_loglik, theta0)
mu = res.x[0]
sigma = np.exp(res.x[1])
nu = np.exp(res.x[2])

pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma],
    "nu": [nu]
}).to_csv("testout7_2.csv", index=False)
