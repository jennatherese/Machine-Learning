import pandas as pd
titanic=pd.read_csv('Titanic-Dataset.csv')
columns_to_drop=['PassengerId','Name','Ticket','Cabin']
titanic_clean=titanic.drop(columns=columns_to_drop)
titanic_encoded=pd.get_dummies(titanic_clean,columns=['Sex','Embarked'])
print(titanic_encoded.head().to_string())