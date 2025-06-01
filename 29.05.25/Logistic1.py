import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("titanic.csv")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True) 
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
features = ['Age', 'Fare', 'Pclass', 'Sex_male']
X = df[features]
y = df['Survived']
X = X.dropna()
y = y.loc[X.index]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(confusion_matrix(y_test, y_pred))
print( classification_report(y_test, y_pred))
