import pandas as pd
df = pd.read_parquet(r"C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp\fg_project\artifacts\features_fg.parquet")
print(df.columns.tolist()[:120])