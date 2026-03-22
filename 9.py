from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV

data = load_iris()
X, y = data.data, data.target

# Przetestuj na innym zbiorze danych, np digits
# Zaobserwuj jak długo wyszukuje hiperparametry...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_distributions = {
    'n_estimators' : range(10,511,50),
    'max_depth' : [3, 4, 5, 6, None]
}

model = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,
    cv=10,
    scoring='accuracy',
    random_state=42)
grid_search.fit(X_train, y_train)

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepsza dokładność: {grid_search.best_score_}")
