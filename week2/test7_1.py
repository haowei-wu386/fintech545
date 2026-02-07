import pandas as pd
import numpy as np

x = pd.read_csv("test7_1.csv").iloc[:, 0].to_numpy()

mu_hat = np.mean(x)
sigma_hat = np.sqrt(np.mean((x - mu_hat)**2))

pd.DataFrame({
    "mu": [mu_hat],
    "sigma": [sigma_hat]
}).to_csv("testout7_1.csv", index=False)
