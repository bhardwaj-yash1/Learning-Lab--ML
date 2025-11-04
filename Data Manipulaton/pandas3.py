#feature engineering : creating new features from the existing raw data that makes model smarter
import pandas as pd
import numpy as np
print("\n")
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 45, 35, 50, 29],
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore', 'Delhi'],
    'joining_date': ['2020-05-01', '2018-07-15', '2019-06-20', '2021-01-01', '2017-03-12'],
    'salary': [50000, 80000, 60000, 90000, 45000],
    'experience_years': [2, 10, 5, 1, 7]
})
# extracting features from joining date

df['joining_date']=pd.to_datetime(df['joining_date'])
df['join_year']=df['joining_date'].dt.year
df['join_month']=df['joining_date'].dt.month
df['join_day']=df['joining_date'].dt.day

# creating new features

df['days_since_joining']=(pd.Timestamp.today()-df['joining_date']).dt.days
print(df)
print("\n")

#binning and bucketing

print("binning")
df['age_group']=pd.cut(df['age'],bins=[0,30,40,60],labels=['young','mid','senior'])
print(df)
print("\n")

#creating new features from existing ones

df['salary_per_year']=df['salary']/df['experience_years']
print(df)
print("\n")

#data encoding ,making dummy columns and handling multicollinearity 
# text to numbers
df_encoded = pd.get_dummies(df, columns=['city', 'age_group'], drop_first=True)
print(df_encoded)

#  scaling and normalization : used when the numerical datas in our
#  dataset vary too much in magnitude (not considering the unit)
#  the models like decision trees gets biased towards higher results
#  and model might get trained inappropriately

from sklearn.preprocessing import MinMaxScaler
 #scales value from 0 to 1 using a formula
# df[['salary_scaled', 'experience_scaled']] = scaler.fit_transform(df[['salary', 'experience_years']])
scaler = MinMaxScaler()
df[['salary_scaled','experience_scaled']]=scaler.fit_transform(df[['salary','experience_years']])
print(df)
print("\n")

 # detecting skew and handling it

print(df['salary'].skew())
df['log_salary']=np.log1p(df['salary'])
print(df)
print("\n")

# polynmial features :
# polynomial features are new features created by rAISING THE CURRENT features to power
# we need them when the data is given in form of powers of features so we manually ad these as new features in the data

df['age_squared']=df['age']**2
print(df)
print("\n")

#mapping or label encoding : number to  text 

df['experience_level']=df['experience_years'].map(lambda x:'junior' if x<3 else 'senior')
print(df)
print("\n")
# count encoding 
# maping the cloumn with its valuecounts
city_counts=df['city'].value_counts()
df['city_freq']=df['city'].map(city_counts)
print(df)
print("\n")
