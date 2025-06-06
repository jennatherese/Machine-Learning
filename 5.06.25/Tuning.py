from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
data = load_wine()
X = data.data
y = data.target
np.random.seed(42)
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]
k = 5
fold_size = len(X) // k
accuracies = []
for i in range(k):
    start = i * fold_size
    end = (i + 1) * fold_size if i < k - 1 else len(X)
    X_test = X_shuffled[start:end]
    y_test = y_shuffled[start:end]
    X_train = np.concatenate((X_shuffled[:start], X_shuffled[end:]), axis=0)
    y_train = np.concatenate((y_shuffled[:start], y_shuffled[end:]), axis=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {i+1} Accuracy: {acc:.4f}")
avg_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy {avg_accuracy:.4f}")
