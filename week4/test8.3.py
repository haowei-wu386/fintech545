import pandas as pd
import numpy as np
from scipy.stats import t

# Read return series
df = pd.read_csv("test7_2.csv")
x1 = df["x1"].dropna()

# Fit Student's t distribution using maximum likelihood estimation
df_hat, loc_hat, scale_hat = t.fit(x1)

# Generate simulated returns from fitted t distribution
N = 1000000
rng = np.random.default_rng(42)
sim = t.rvs(df_hat, loc=loc_hat, scale=scale_hat, size=N, random_state=rng)

# Simulation VaR (95% => left tail 5%) 
q05 = np.quantile(sim, 0.05)

var_absolute_sim = -q05
var_diff_mean_sim = sim.mean() - q05

# Analytical VaR from the fitted t distribution (for comparison with simulation) 
q05_param = t.ppf(0.05, df_hat) * scale_hat + loc_hat
var_absolute_82 = -q05_param
var_diff_mean_82 = loc_hat - q05_param  # = -(q05_param - loc_hat)

# Write results to CSV file
out = pd.DataFrame({
    "VaR Absolute": [var_absolute_sim],
    "VaR Diff from Mean": [var_diff_mean_sim]
})
out.to_csv("testout_8.3.csv", index=False)

print("\n--- Compare Simulation vs 8.2 (Parametric) ---")
print("8.2 VaR Absolute       :", var_absolute_82)
print("Sim VaR Absolute       :", var_absolute_sim)
print("Diff                  :", var_absolute_sim - var_absolute_82)

print("8.2 VaR Diff from Mean :", var_diff_mean_82)
print("Sim VaR Diff from Mean :", var_diff_mean_sim)
print("Diff                  :", var_diff_mean_sim - var_diff_mean_82)
