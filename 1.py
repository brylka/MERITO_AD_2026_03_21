from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

random_forest = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_split=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42)
random_forest.fit(X_train, y_train)

train_acc = single_tree.score(X_train, y_train)
test_acc = single_tree.score(X_test, y_test)

print("POJEDYNCZE DRZEWO DECYZYJNE")
print("-" * 40)
print(f"Dokładność na zbiorze treningowym: {train_acc:.4f}")
print(f"Dokładność na zbiorze testowym:    {test_acc:.4f}")

cv_scores = cross_val_score(single_tree, X, y, cv=10)
print(f"Walidacja cv:")
print(f"Średnia dokładność: {cv_scores.mean():.4f}")
print(f"Odchylenie std:     {cv_scores.std():.4f}")


random_forest_train_acc = random_forest.score(X_train, y_train)
random_forest_test_acc = random_forest.score(X_test, y_test)

print("LOSOWY LAS")
print("-" * 40)
print(f"Dokładność na zbiorze treningowym: {random_forest_train_acc:.4f}")
print(f"Dokładność na zbiorze testowym:    {random_forest_test_acc:.4f}")

random_forest_cv_scores = cross_val_score(random_forest, X, y, cv=10)
print(f"Walidacja cv:")
print(f"Średnia dokładność: {random_forest_cv_scores.mean():.4f}")
print(f"Odchylenie std:     {random_forest_cv_scores.std():.4f}")