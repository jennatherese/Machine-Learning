import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
titanic = pd.read_csv('Titanic-Dataset.csv')
numerical_cols = titanic.select_dtypes(include=['int64', 'float64'])
corr_matrix = numerical_cols.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", color="black")

plt.title("Correlation Matrix ")
plt.tight_layout()
plt.show()