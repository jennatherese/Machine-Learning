import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Titanic-Dataset.csv")
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

X = df[['Age']].values  
y = df['Fare'].values   

X_b = np.c_[np.ones((X.shape[0], 1)), X]  
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
y_pred = X_b.dot(theta_best)


plt.scatter(X, y, color='gray', label='Actual Fare')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Linear Regression (Fare ~ Age)')
plt.legend()
plt.show()
print("Intercept (θ₀):", theta_best[0])
print("Slope (θ₁):", theta_best[1])
