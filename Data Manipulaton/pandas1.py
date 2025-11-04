import pandas as pd
import pandas as pd

# Load sample dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
print(df)
print(df.head())
print("\n")
print(df.shape)
print("\n")
print(df.columns)
print("\n")
print(df.info())
print("\n")
print(df.tail())
print("\n")
print(df.describe())
print("\n")
print("\n")
print(df['tip'])
print("\n")
print([['tip']])
print("\n")
print(df.iloc[[1,5],[0,2]]) # iloc for index
print("\n")
print(df.loc[[1,5],['tip','sex']]) 
print("\n")# loc for accessing using labels
print(df.iloc[4]) # nth row by columns labels
print("\n")
print("\n")
print(df['tip']>4)
print("\n")
print(df['sex']=="Female")
print("\n")
print("adding and modifying columns")
df['tip_percent']=(df['tip']/df['total_bill'])*100
print(df)
print("\n")
df.drop('tip_percent',axis=1,inplace=True) # axis 1 is for columns
# df.loc[2] = ['Charlie', 35, 70000] to add rows, syntax
#can be done using pd.concat()
print("\n")
print("accessing and modifying data types")
print(df.dtypes)
print("\n")
df['size']=df['size'].astype('float')
df['sex']=df['sex'].astype('category')
print(df.dtypes)

print("working with series - column level operations")

print(df['tip'].value_counts())
#takes all the unique values of this columns and tells how many times they repeat

print("\n")
print(df['tip'].mean())
print(df['total_bill'].mean())

print(df.isnull().sum())
# return total number of null values columns wise