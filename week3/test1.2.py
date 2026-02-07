import pandas as pd

df = pd.read_csv("test1.csv")

# Skip the first line
df_numeric = df.select_dtypes(include="number")

# Skip missing rows
df_clean = df_numeric.dropna()

corr_matrix = df_clean.corr()
corr_matrix.to_csv("testout_1.2.csv", index=False)