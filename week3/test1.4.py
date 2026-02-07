import pandas as pd

df = pd.read_csv("test1.csv")

# Skip the first line
df_numeric = df.select_dtypes(include="number")

# Pairwise skip missing
corr_matrix = df_numeric.corr()

corr_matrix.to_csv("testout_1.4.csv", index=False)