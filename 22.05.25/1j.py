import pandas as pd
titanic = pd.read_csv('Titanic-Dataset.csv')
titanic_clean = titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
titanic_clean['Age'].fillna(titanic_clean['Age'].median(), inplace=True)
titanic_clean['Embarked'].fillna(titanic_clean['Embarked'].mode()[0], inplace=True)
titanic_clean = pd.get_dummies(titanic_clean, columns=['Sex', 'Embarked'])
X = titanic_clean.drop(columns=['Survived'])  
y = titanic_clean['Survived']                


X.to_csv('titanic_features.csv', index=False)
y.to_csv('titanic_target.csv', index=False)


titanic_clean.to_csv('titanic_cleaned.csv', index=False)

print("Processing complete!")
print(f"Original shape: {titanic.shape}")
print(f"Cleaned features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("\nFirst 3 rows of cleaned data:")
print(titanic_clean.head(3))