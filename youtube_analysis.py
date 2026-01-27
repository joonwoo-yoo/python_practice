# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
sample_index = np.random.choice(len(data))
print(f"Sample Index : {sample_index}")
data.iloc[sample_index]

# %%
data.isnull()

# %%
data.isnull().sum()     # axis=0 (default) : column-wise / axis=1 : raw-wise
data.isnull().sum(axis=1)

# %%

sample_indicies_with_nan = data.isnull().sum(axis=1) > 0
data[sample_indicies_with_nan]

# %%
# 122~123개의 행에 NaN이 동일하게 포함되어 있으므로, 8개의 변수 값이 공통으로 비어있을 거라는 가설
sample_indicies_with_nan = data.isnull().sum(axis=1) >= 8
data_with_nans = data[sample_indicies_with_nan]
print(len(data_with_nans))
data_with_nans

# %%
columns_with_nan = [
    "Country",
    "Abbreviation",
    "Population",
    "Gross tertiary education enrollment (%)",
    "Unemployment rate",
    "Urban_population",
    "Latitude",
    "Longitude"
]
# 위에 선택한 8개 변수들이 가설대로 모두 NaN인지 확인
data_with_nans[columns_with_nan].isnull().all(axis=1).sum()

# %%
# 음의 값만을 가지는 변수 내에 음수 값이 있는지 확인
(data["subscribers"] < 0).any()

# %%
print(data[["Latitude", "Longitude"]].min())
print(data[["Latitude", "Longitude"]].max())

# %%
data.describe()

# %%
target_feature = "lowest_monthly_earnings"
sns.boxplot(y=target_feature, data=data)
plt.title(f"{target_feature}", pad=10)
plt.ylabel("")
plt.show()

# %%
# 이상치 제거를 위한 분위수 확인 (유튜버 분석이기에 이상치는 언제든 존재할 수있음)
Q3 = data["lowest_monthly_earnings"].quantile(q=0.75)
Q1 = data["lowest_monthly_earnings"].quantile(q=0.25)
IQR = Q3 - Q1
print(f"[Quantile 25% = {Q1}] \n[Quantile 75% = {Q3}] \n[IQR = {IQR}]")

# %%
sns.histplot(data=data, x=target_feature, kde=True)
plt.show()

# %%
sns.histplot(data=data, x=target_feature, log_scale=(False, True))     # 출력이 왜 안되는지 확인
plt.show()

# %%
