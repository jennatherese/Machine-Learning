from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
def rank_models(models, X_test, y_test, metric):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric == 'f1':
            score = f1_score(y_test, y_pred, average='weighted')
        elif metric == 'roc_auc':
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                score = roc_auc_score(y_test, y_prob, multi_class='ovr')
            else:
                score = float('nan')
        else:
            raise ValueError("Unsupported metric")
        results.append({'Model': name, metric: score})
    df = pd.DataFrame(results)
    return df.sort_values(by=metric, ascending=False).reset_index(drop=True)
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier()
log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
models = {
    'Logistic Regression': log_model,
    'Decision Tree': tree_model}
result = rank_models(models, X_test, y_test, metric='accuracy')
print(result)
