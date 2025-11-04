# EDA - EXPLOARATORY DATA ANALYSIS
# -- the art of asking data the right questions
# goal of eda :
# identif : patterns in data , distribution of features , 
# outliers or bad data, correlations between variables, 
# which features are most predictive

import pandas as pd

df = pd.DataFrame({
    'age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 48],
    'salary': [25000, 27000, 52000, 61000, 58000, 80000, 79000, 82000, 87000, 60000],
    'gender': ['F', 'M', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'F'],
    'purchased': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
})

print(df)
print(df.info())

#2 checking missing values 

print(df.isnull().sum())
print("\n")

# #3 distribution of features

# import matplotlib.pyplot as plt
# df.hist(bins=10, figsize=(10, 6))
# plt.show()

# print(df['purchased'].skew())
# print(df['age'].skew())

# import seaborn as sns

# sns.histplot(df['salary'], kde=True)
# plt.show()


# # 4 categorial value counts : counts frequency of unique value in the mentioned column

# print(df['gender'].value_counts(normalize=False))
# print(df['purchased'].value_counts(normalize=True))
# print(df)

# # 5 outlier detection


# # the outliers will be seen outside of box as dots
# sns.boxplot(x=df['salary'])
# plt.show()

# #6 relationships bewtween features : numerical value holders

# print(df['salary'].skew())

# # cross plots all the features which reveals how the relationship of features effects the purchase

# sns.pairplot(df, hue='purchased')
# plt.show()

# # using correlation heatmap:
# # ---- correlation tells us how closely two numerical features are related
# # +1 → Perfect positive correlation
# # (As one increases, the other increases)

# # -1 → Perfect negative correlation
# # (As one increases, the other decreases)

# #  0 -> no correlation

# # A correlation heatmap is a colorful table 
# # showing correlation coefficients between all 
# # numeric features in your dataset.

# corr = df.corr(numeric_only=True)
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()

# # GROUPED ANALYSIS : split your data in groups based 
# # in one or more features,then apply aggregations 
# # to analyze each group individually

print(df.groupby('gender')['salary'].mean())
# isko hum pehle group karne k liye use karte ek feature ko dusre feature k basis par
# like we are grouping salary here and the display its aggregates

print(df.groupby('gender')['salary'].count())
print(df.groupby('gender')['purchased'].mean())

print(df.groupby('gender')['salary'].max())
print(df.groupby('salary')['gender'].max())
print(df.groupby(['gender', 'purchased'])['salary'].count())
# unkki salary jin genders ne purchase kiya hai ya nahi kiya 
# yaha grouping salary ki hoti hai

# 8 TARGET VALUE ANALYSIS :


# target variable vo variable hai jiske 
# baare mai hum predictions karege using our  dataset

# target vallue analysis tells us how the
#  target variable behaves :
# accross different features,
# for class imbalance, == occuring of one class or category a lot more than the other countplot is used to detect
# for feature relevance,== how much the feature contributes to the prediction of target value find using correlation
# to guide feature engineering == 
# and modeling strategy
import seaborn as sns
import matplotlib.pyplot  as plt
sns.boxplot(x='purchased', y='salary', data=df)

plt.show()