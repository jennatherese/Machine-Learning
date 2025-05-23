import pandas as pd 
df=pd.read_csv("Titanic-Dataset.csv")
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(columns=['Cabin'],inplace=True)
print("missing values after handling:")
print(df.isnull().sum())


