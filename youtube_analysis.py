# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as D
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
# Q3 = data["lowest_monthly_earnings"].quantile(q=0.75)
# Q1 = data["lowest_monthly_earnings"].quantile(q=0.25)
# IQR = Q3 - Q1
# print(f"[Quantile 25% = {Q1}] \n[Quantile 75% = {Q3}] \n[IQR = {IQR}]")

# %%
sns.histplot(data=data, x=target_feature, kde=True)
plt.show()

# %%
sns.histplot(data=data, x=target_feature, log_scale=(False, True))     # 출력이 왜 안되는지 확인
plt.show()

# %%
category_feature = "category"                   # x 축
target_feature = "lowest_monthly_earnings"      # y 축

barplot = sns.barplot(data=data, x=category_feature, y=target_feature, color="C0", errorbar=None)
loc, labels = plt.xticks()
barplot.set_xticklabels(labels, rotation=90)
plt.title("Lowest Monthly Earnings per Each Category", pad=15)
plt.show()

# 변수 간 관계 확인
# %%
corr = data.corr(numeric_only=True)
mask = np.ones_like(corr, dtype=bool)
mask = np.triu(mask)

plt.figure(figsize=(13, 10))
sns.heatmap(data=corr, annot=True, fmt=".2f", mask=mask, linewidths=.5, cmap="RdYlBu_r")
plt.title("Correlation Matrix")
plt.show()

# %%
indices_to_keep = data.isnull().sum(axis=1) < 8

data_original = data
data = data[indices_to_keep].copy()
print(f"데이터 샘플 수 변화 = {len(data_original)} -> {len(data)}")
data["subscribers_for_last_30_days"].fillna(0, inplace=True)
data.dropna(axis=0, inplace=True)
print(f"처리 완료 후 데이터 샘플 수 = {len(data)}")

data.isnull().sum()

# %%
data
# %%
target_feature = "lowest_monthly_earnings"
data[f'log_{target_feature}'] = scale(np.log(data[target_feature] + 1))
data[f'log_{target_feature}'].describe()

# %%
data[[target_feature, f"log_{target_feature}"]].describe()
# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

sns.histplot(data=data, x=target_feature, kde=True, ax=ax[0])
ax[0].set_title("Before Log Transformation")

sns.histplot(data=data, x=f'log_{target_feature}', kde=True, ax=ax[1])
ax[1].set_title("After Log Transformation")

plt.show()

# %%
target_feature = "lowest_monthly_earnings"
standard_scaler = StandardScaler()
data[f"standardized_{target_feature}"] = standard_scaler.fit_transform(data[[target_feature]])

feature_original = data[target_feature]
feature_standardized = data[f"standardized_{target_feature}"]

print(f"평균(mean) 비교 = {feature_original.mean():.7} --> {feature_standardized.mean():.7}")
print(f"표준편차(std) 비교= {feature_original.std():.7} --> {feature_standardized.std():.7}")

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

sns.histplot(data=data, x=target_feature, kde=True, ax=ax[0])
ax[0].set_title("Before Standardization")

sns.histplot(data=data, x=f"standardized_{target_feature}", kde=True, ax=ax[1])
ax[1].set_title("After Standardization")

plt.show()

# %%
target_feature = "lowest_monthly_earnings"
normalized_scaler = MinMaxScaler()
data[f"normalized_{target_feature}"] = normalized_scaler.fit_transform(data[[target_feature]])

feature_original = data[target_feature]
feature_normalized = data[f"normalized_{target_feature}"]

print(f"최소값(min) 비교 = {feature_original.min():.7} --> {feature_normalized.min():.7}")
print(f"최대값(max) 비교 = {feature_original.max():.7} --> {feature_normalized.max():.7}")

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

sns.histplot(data=data, x=target_feature, kde=True, ax=ax[0])
ax[0].set_title("Before Normalization")

sns.histplot(data=data, x=f"normalized_{target_feature}", kde=True, ax=ax[1])
ax[1].set_title("After Normalization")

plt.show()