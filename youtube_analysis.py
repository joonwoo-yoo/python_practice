# %%
import pandas as pd
import numpy as np
import sklearn.datasets as D

# %%
data = pd.read_csv("./data/Global YouTube Statistics.csv", encoding="latin1")
data

# %%
data.columns

# %%
data.info()

# %%
np.random.seed(1234)

# %%
# feature_index = 6
feature_index = np.random.choice(len(data.columns))
print(f"Feature # : {feature_index}")
feature_name = data.columns[feature_index]
print(f"Feature's Name : {feature_name}")

data[feature_name]

# %%
feature_name = "channel_type"
unique_values = data[feature_name].unique()
print("Category :", unique_values)
print("# of Categories :", len(unique_values))

# %%
data[feature_name].value_counts()

# %%
