import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("auto-mpg.csv")
df['horsepower'].replace('?', np.nan, inplace=True)  
df['horsepower'] = pd.to_numeric(df['horsepower'])   
df.dropna(subset=['horsepower'], inplace=True)       

X = df[['horsepower']].values
y = df['mpg'].values
def fit_and_plot_poly(degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    sort_idx = X.flatten().argsort()
    X_sorted = X[sort_idx]
    y_sorted = y_pred[sort_idx]

    plt.plot(X_sorted, y_sorted, label=f'Degree {degree}')
    return mse, r2
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Actual Data')
mse1, r21 = fit_and_plot_poly(1)
mse2, r22 = fit_and_plot_poly(2)
mse3, r23 = fit_and_plot_poly(3)

plt.title("MPG vs Horsepower (Polynomial Regression)")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.grid(True)
plt.show()

print("\n📊 Model Performance:")
print(f"Linear (Degree 1)    MSE: {mse1:.2f}, R²: {r21:.2f}")
print(f"Quadratic (Degree 2) MSE: {mse2:.2f}, R²: {r22:.2f}")
print(f"Cubic (Degree 3)     MSE: {mse3:.2f}, R²: {r23:.2f}")
