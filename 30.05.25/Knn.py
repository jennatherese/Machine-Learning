import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = load_iris(as_frame=True)
df = iris.frame  
X = df.drop(columns=['target'])
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
k_range = range(1, 21)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    print(f"k = {k}, Accuracy = {acc:.4f}")
best_k = k_range[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print(best_k)
print( round(best_accuracy * 100, 2), "%")
plt.figure(figsize=(10, 5))
plt.plot(k_range, accuracies, marker='o', color='green')
plt.title("Validation Accuracy vs Number of Neighbors")
plt.xlabel("Number of Neighbors")
plt.ylabel("Validation Accuracy")
plt.xticks(k_range)
plt.grid(True)
plt.show()
