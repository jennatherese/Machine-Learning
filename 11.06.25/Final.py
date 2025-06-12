from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_configs = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {'model__C': [0.01, 0.1, 1, 10]}},
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {'model__max_depth': [3, 5, 10, None]}},
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10, None]}},
    'SVM': {
        'model': SVC(probability=True),
        'params': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}}
}
results = []
for name, config in model_configs.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', config['model'])
    ])
    grid = GridSearchCV(pipe, config['params'], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Best Params': grid.best_params_,
        'F1 Score': f1,
        'AUC-ROC': auc
    })
leaderboard = pd.DataFrame(results)
leaderboard = leaderboard.sort_values(by=['F1 Score', 'AUC-ROC'], ascending=False).reset_index(drop=True)
print(leaderboard)
