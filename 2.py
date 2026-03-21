from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

for n_trees in [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]:
    random_forest = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=42)
    random_forest.fit(X_train, y_train)

    print(f"Ilość drzew: {n_trees}:")
    print(f"Dokładność na zbiorze treningowym: {random_forest.score(X_train, y_train):.4f}")
    print(f"Dokładność na zbiorze testowym:    {random_forest.score(X_test, y_test):.4f}")
    print(f"OOB score:                         {random_forest.oob_score_:.4f}")
