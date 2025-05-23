import pandas as pd
titanic=pd.read_csv('Titanic-Dataset.csv')
titanic_encoded=pd.get_dummies(titanic,columns=['Sex','Embarked'])
necessary_columns=['PassengerId','Survived','Pclass','Age','SibSp','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']
titanic_final=titanic_encoded[necessary_columns]
print(titanic_final.head().to_string())