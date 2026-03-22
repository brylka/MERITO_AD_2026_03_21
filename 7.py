from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

data = load_iris()
X, y = data.data, data.target

# Przetestuj na innym zbiorze danych, np digits
# Zaobserwuj jak długo wyszukuje hiperparametry...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators' : [10, 60, 110],
    'max_depth' : [3, 4, 5]
#    'n_estimators' : [10,60,110,160,210,260,310,360,410,460,510],
#    'max_depth' : [3, 4, 5, 6, None]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepsza dokładność: {grid_search.best_score_}")
