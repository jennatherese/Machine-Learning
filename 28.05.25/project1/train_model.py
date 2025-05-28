import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
df = pd.read_csv("student-mat.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

features = ['G1', 'G2', 'studytime', 'failures', 'absences', 'goout']
data = df_encoded[features + ['G3']].copy()
data['avg_grade'] = (data['G1'] + data['G2']) / 2
data['engagement_score'] = data['studytime'] - data['goout']
data['absences'] = np.clip(data['absences'], 0, 30)
data['failures'] = np.clip(data['failures'], 0, 3)
X = data.drop(columns=['G3'])
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Save the trained model
joblib.dump(lr_model, "final_model.pkl")

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 Evaluation:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# (Optional) Plot Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([0, 20], [0, 20], color='red')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grade")
plt.grid(True)
plt.show()
