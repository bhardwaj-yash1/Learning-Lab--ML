# stage : 2 -- data cleaning and data preprocessing 


import pandas as pd
import numpy as np

# Sample dirty dataset
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice'],
    'age': [25, 32, np.nan, 45, 29, 25],
    'income': [50000, 54000, 61000, np.nan, 48000, 50000],
    'city': ['Delhi', 'delhi', 'Mumbai', 'Bangalore', np.nan, 'Delhi'],
    'smoker': ['No', 'yes', 'Yes', 'no', 'No', 'No']
}

df = pd.DataFrame(data)
print(df)

print(df.isnull().sum())

#df.dropna() drops all teh rows with even one nan value

df.fillna({'age':30},inplace=True)
print(df)
print("\n")
df.fillna({'income':df['income'].mean()},inplace=True)
print(df)
print("\n")
df['city']=df['city'].str.lower()

print(df['city'])
print("\n")
# df['smoker']=df['smoker'].str.lower()
df['smoker']=df['smoker'].replace({'Yes':'yes','No':'no'})
print(df['smoker'])
print(df.duplicated())
#to drop duplicate rows : 
df.drop_duplicates(inplace=True)
print(df)
print("\n")
df.rename(columns={'income':'monthly_income'},inplace=True)
print(df)
print('\n')
print("creating new columns ")
df['income_per_age']=df['monthly_income']/df['age']
print(df)
print("\n")
#  pd.get_dummies() → Converts categorical columns(text data, like category mumbei , categoory delhi) into numerical (binary) columns
# ✅ Each unique value in the column becomes a new column (one-hot encoding)
# ✅ The original column is replaced

df_encoded = pd.get_dummies(df, columns=['city', 'smoker'], drop_first=True)
print(df_encoded)

print("outlier detection")
Q1 = df['monthly_income'].quantile(0.25)
Q3 = df['monthly_income'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['monthly_income'] < Q1 - 1.5 * IQR) | (df['monthly_income'] > Q3 + 1.5 * IQR)]
print(outliers)