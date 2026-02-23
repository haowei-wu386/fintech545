import numpy as np
import pandas as pd
from scipy.stats import t

from scipy.integrate import quad

alpha = 0.95

df = pd.read_csv("test7_2.csv") 
returns = df["x1"]

loss = -returns
mean_loss = loss.mean()

# Fit t distribution
df_t, loc_t, scale_t = t.fit(loss)

# Number of Monte Carlo simulations
n_sims = 200000
rng = np.random.default_rng(42)

# Simulate losses from fitted t distribution
sim_losses = t.rvs(df_t, loc=loc_t, scale=scale_t, size=n_sims, random_state=rng)

# Compute VaR threshold (alpha-quantile of simulated losses)
q_sim = np.quantile(sim_losses, alpha) 

# Expected Shortfall (ES): average of worst losses beyond VaR
es_absolute = sim_losses[sim_losses >= q_sim].mean()

es_diff_mean = es_absolute - mean_loss

result = pd.DataFrame({
    "ES Absolute": [es_absolute],
    "ES Diff from Mean": [es_diff_mean]
})

result.to_csv("testout8_6.csv", index=False)
