import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(columns=['Cabin'],inplace=True)
df['FamilySize']=df['SibSp']+df['Parch']+1
df['Title']=df['Name'].str.extract('([A-Za-z]+)\.',expand=False)
print(df[['Name','SibSp','Parch','FamilySize','Title']].head(10))