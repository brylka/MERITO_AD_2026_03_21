import xgboost as xdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

random_forest = xdb.XGBClassifier(
    n_estimators=150,
    max_depth=None,
    learning_rate=0.01,
    random_state=42)
random_forest.fit(X_train, y_train)

random_forest_train_acc = random_forest.score(X_train, y_train)
random_forest_test_acc = random_forest.score(X_test, y_test)

print("XGBOOST:")
print("-" * 40)
print(f"Dokładność na zbiorze treningowym: {random_forest_train_acc:.4f}")
print(f"Dokładność na zbiorze testowym:    {random_forest_test_acc:.4f}")

random_forest_cv_scores = cross_val_score(random_forest, X, y, cv=10)
print(f"Walidacja cv:")
print(f"Średnia dokładność: {random_forest_cv_scores.mean():.4f}")
print(f"Odchylenie std:     {random_forest_cv_scores.std():.4f}")