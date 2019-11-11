# Preamble...

import numpy as np
import pandas as pd

df = pd.read_csv("SouthAfrica.csv")

# Standardize data
df2 = df.copy()
df2["famhist"] = 1 * (df2["famhist"] == "Present")
cols = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]
df2[cols] = (df2[cols] - df2[cols].mean(0)) / df2[cols].std(0)
df2 = df2.drop("row.names", 1)


# Data
X = df2[cols].values
y = df2.values[:,-1]


y_regr = df2["typea"].values.squeeze()
X_regr = df2.drop(columns="typea").values.squeeze()

print(np.sum((y==1)*1))
