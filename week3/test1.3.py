import pandas as pd

df = pd.read_csv("test1.csv")

# Skip the first line
df_numeric = df.select_dtypes(include="number")

# Pairwise skip missing
cov_matrix = df_numeric.cov()

cov_matrix.to_csv("testout_1.3.csv", index=False)